# Interpretabilità differenziabile in DINOv2: gradienti, token e rappresentazioni spaziali

**Data di creazione:** 29 luglio 2026

## Scopo della nota

Questa nota conserva il ragionamento di machine learning dietro un problema di interpretabilità emerso con DINOv2: perché una predizione può essere calcolata correttamente mentre Grad-CAM non può farlo, e perché una rete Transformer richiede un passaggio esplicito dalla sequenza di token a una rappresentazione spaziale.

Non è una guida di modifica del codice. Il caso concreto riguarda `DinoV2B14` e una classificazione multi-label di ingredienti, ma i principi si applicano a qualsiasi Vision Transformer (ViT) usato con metodi di saliency basati sui gradienti.

## Predizione e spiegazione sono compiti diversi

Una predizione in inferenza valuta una funzione:

$$y = f_\theta(x),$$

dove $x$ è l'immagine, $\theta$ sono i parametri del modello e $y$ sono i logit. Per classificazione multi-label, ogni componente $y^c$ è il punteggio indipendente di una classe $c$; non c'è una softmax che renda le classi mutuamente esclusive.

Una spiegazione Grad-CAM non richiede soltanto $y^c$. Richiede come quel logit cambia al variare di una rappresentazione intermedia $A$:

$$\frac{\partial y^c}{\partial A}.$$

La predizione è quindi un problema di *forward pass*; Grad-CAM è un problema di forward pass più backward pass. È possibile che il primo sia perfettamente valido mentre il secondo sia impossibile o non definito nel grafo di autograd costruito per quell'inferenza.

## Grad-CAM come proiezione del gradiente su una mappa di feature

Per una CNN, si sceglie tipicamente un layer con attivazioni:

$$A \in \mathbb{R}^{B \times C \times H \times W}.$$

Per una classe $c$, Grad-CAM calcola un peso per canale facendo la media spaziale del gradiente:

$$\alpha_k^c = \frac{1}{HW}\sum_{i=1}^{H}\sum_{j=1}^{W}
\frac{\partial y^c}{\partial A_{kij}}.$$

La mappa di rilevanza è poi:

$$L_{\mathrm{GradCAM}}^c = \operatorname{ReLU}\left(\sum_{k=1}^{C}\alpha_k^c A_k\right).$$

L'intuizione è precisa: un canale riceve un peso alto quando aumentare le sue attivazioni aumenta il logit della classe osservata. La combinazione pesata conserva l'indice spaziale $(i,j)$, dunque può essere sovrapposta all'immagine.

Questa definizione implica due requisiti:

1. deve esistere un gradiente del logit rispetto al layer scelto;
2. le attivazioni devono poter essere interpretate come mappa spaziale, direttamente oppure tramite una trasformazione nota.

## Congelamento, autograd e gradienti delle attivazioni

Congelare un backbone significa impostare `requires_grad=False` sui suoi parametri. In addestramento, questa scelta impedisce che il gradiente venga accumulato nei pesi congelati e che l'ottimizzatore li modifichi.

Se, inoltre, l'input $x$ non richiede gradiente, PyTorch non ha motivo di costruire un grafo differenziabile per le operazioni del backbone. Le sue attivazioni intermedie risultano quindi non differenziabili rispetto al logit, pur essendo numericamente disponibili.

Questo non è un errore del modello: è l'ottimizzazione corretta per una normale inferenza. Diventa un limite soltanto quando un metodo esplicativo necessita del backward pass.

È utile distinguere i concetti:

| Concetto | Significato |
| --- | --- |
| Parametro congelato | Il peso non riceve gradiente e non viene aggiornato. |
| Input differenziabile | È possibile calcolare come l'output cambia rispetto all'input e alle attivazioni lungo il percorso. |
| Gradiente per Grad-CAM | È un gradiente temporaneo usato per stimare rilevanza, non un aggiornamento dei pesi. |

Rendere l'input differenziabile basta a riaprire il percorso di gradienti attraverso operatori con pesi congelati. Questo permette di calcolare $\partial y^c / \partial A$ senza trasformare il modello in un modello allenabile e senza eseguire un aggiornamento di ottimizzazione.

## Dalla griglia delle patch alla sequenza Transformer

Una CNN mantiene esplicitamente assi spaziali `H×W` lungo la rete. Un ViT li trasforma invece in una sequenza.

Con patch di lato $P=14$ e immagine quadrata $H=W=224$, il numero di patch è:

$$N = \frac{H}{P}\frac{W}{P} = 16 \cdot 16 = 256.$$

Ogni patch viene proiettata in un embedding di dimensione $D=768$. Il backbone lavora quindi su patch token:

$$E \in \mathbb{R}^{B \times N \times D}.
$$

DINOv2 ViT-B/14 con registers aggiunge alla sequenza un class token e quattro register token:

$$Z \in \mathbb{R}^{B \times (1+4+256) \times 768}
= \mathbb{R}^{B \times 261 \times 768}.$$

Il class token è un vettore appreso usato per aggregare informazione globale. I register token sono slot di memoria appresi: non corrispondono a regioni dell'immagine e aiutano il modello a non usare patch di sfondo come memoria di lavoro. Entrambi partecipano alla self-attention, ma non hanno una coordinata $(i,j)$ nella griglia di patch.

## Perché una sequenza non è una heatmap

Un layer Transformer restituisce normalmente un tensore `[B, T, D]`, con $T$ token. Grad-CAM classico, invece, assume canali e coordinate: `[B, C, H, W]`.

Per ottenere una saliency map da un ViT bisogna esplicitare la corrispondenza:

```text
[B, 261, 768]
  → rimuovi class token e register token
[B, 256, 768]
  → 256 = 16 × 16
[B, 768, 16, 16]
```

La trasformazione non è una semplice convenzione di shape: dichiara che ciascuno dei 256 token rimanenti rappresenta una precisa patch dell'immagine. Dopo il reshape, le 768 dimensioni di embedding diventano i canali della feature map e Grad-CAM può mediare i gradienti sulle 256 posizioni.

Se class token o register token fossero inclusi nel reshape, verrebbero assegnati artificialmente a una posizione dell'immagine e la heatmap non avrebbe più una semantica spaziale corretta.

## Dove agganciare Grad-CAM in un Transformer

Il layer scelto deve soddisfare contemporaneamente tre proprietà:

1. essere abbastanza profondo da codificare informazione semantica rilevante per la classe;
2. trovarsi sul percorso differenziabile verso il logit;
3. esporre token patch che possano essere ricostruiti in griglia.

Nel caso DINOv2, la LayerNorm prima della self-attention dell'ultimo blocco (`norm1`) è un punto naturale: le sue feature entrano nell'attenzione finale e influenzano l'output della rete. La normalizzazione finale del backbone è meno adatta come hook concettuale quando viene invocata più volte per estrarre più livelli intermedi: un singolo modulo può allora produrre più attivazioni nel medesimo forward e rendere ambigua l'associazione fra attivazione, gradiente e livello semantico.

Una heatmap ViT non è identica a una heatmap CNN. Ha risoluzione nativa pari alla griglia delle patch (`16×16` a input 224), perciò l'upsampling a 224×224 rende la visualizzazione più leggibile ma non crea dettaglio informativo nuovo sotto la dimensione di una patch.

## La head `_lc` e le rappresentazioni multi-livello

Il wrapper DINOv2 `_lc` usato nel progetto effettua una classificazione lineare su una rappresentazione composta. Non usa soltanto l'ultimo class token; concatena:

$$h = [c_9; c_{10}; c_{11}; c_{12}; \operatorname{mean}(E_{12})]
\in \mathbb{R}^{3840}.$$

I termini $c_9,\ldots,c_{12}$ sono i class token degli ultimi quattro blocchi; ognuno è un vettore da 768 dimensioni. $\operatorname{mean}(E_{12})$ è la media dei patch token dell'ultimo blocco, anch'essa di dimensione 768. La classificazione effettiva è:

$$y = Wh + b, \qquad W \in \mathbb{R}^{C \times 3840}.$$

Questa costruzione mescola rappresentazioni globali di livelli profondi diversi con una sintesi delle feature distribuite sulle patch. Per un task multi-label, ogni riga di $W$ corrisponde a un ingrediente e produce il relativo logit.

## Deep Feature Factorization e compatibilità dimensionale

DFF applica una fattorizzazione non negativa alle attivazioni spaziali. Se la mappa è `[B, 768, 16, 16]`, ogni concetto ottenuto dalla fattorizzazione vive nello spazio dei canali:

$$z_q \in \mathbb{R}^{768}.$$

Per assegnare etichette ai concetti, DFF applica un classificatore a $z_q$. Qui emerge una differenza fondamentale fra “feature locale” e “input reale della head”: la head `_lc` attende $h \in \mathbb{R}^{3840}$, mentre un concetto patch ha 768 dimensioni.

La porzione della head associata alla media delle patch è:

$$W_{\mathrm{patch}} = W[:, -768:] \in \mathbb{R}^{C \times 768}.$$

La proiezione $W_{\mathrm{patch}}z_q+b$ non riproduce l'inferenza completa: misura soltanto come il concetto patch si allinea con il segmento della decisione finale che dipende dalla sintesi delle patch. È quindi una semantica adatta a etichettare concetti DFF, ma non una sostituzione della head originale.

## Implicazioni interpretative

- Grad-CAM mostra sensibilità locale del logit, non una prova causale che la regione contenga davvero l'ingrediente.
- Una classe può ricevere rilevanza alta su contesto, stoviglie o composizione visiva se tali elementi sono correlati alla classe nei dati di training.
- Per ingredienti non visibili, mescolati o coperti, la saliency non può fornire evidenza visiva diretta; il modello può usare correlazioni apprese.
- La saliency dipende dal target selezionato: due ingredienti predetti dalla stessa immagine possono produrre mappe diverse.
- La spiegazione DFF basata su $W_{\mathrm{patch}}$ descrive il contributo dei concetti patch, non quello completo dei quattro class token.

## Sunto

Grad-CAM richiede gradienti delle attivazioni, mentre una normale inferenza con backbone congelato può eseguire soltanto il forward e non costruire un percorso differenziabile. Un ViT richiede inoltre la riconversione esplicita dei patch token in griglia spaziale, escludendo token globali come CLS e registers.

Nel classificatore DINOv2 `_lc`, la decisione finale usa una feature multi-livello da 3840 dimensioni. Le tecniche basate sui concetti patch operano invece nello spazio da 768 dimensioni delle patch; interpretarle correttamente richiede distinguere il contributo locale delle patch dalla classificazione globale completa.
