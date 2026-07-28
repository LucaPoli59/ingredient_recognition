# Deep dive: DINOv2 ViT-B/14 per la predizione degli ingredienti

**Data di creazione:** 28 luglio 2026

## Scopo e perimetro

`DinoV2B14` (`src/models/dinov2.py`) adatta al problema multi-label il checkpoint upstream `dinov2_vitb14_reg_lc` di `facebookresearch/dinov2`. In questa pipeline non è un modello generativo né viene rieseguito l'obiettivo self-supervised DINO: è un estrattore Vision Transformer già preaddestrato, a cui il repository applica una testa lineare supervisionata per stimare gli ingredienti.

Il forward restituisce un tensore `[B, C]`, dove `B` è il batch e `C = num_classes` è il numero di ingredienti codificati dal DataModule. I valori sono logit indipendenti; la sigmoid è applicata nella loss/valutazione Lightning, non all'interno del modello. Di conseguenza, il modello resta compatibile con `BCEWithLogitsLoss` e con target multi-hot.

## Il modello DINOv2: architettura e rappresentazione

### Quale variante viene caricata

Il progetto usa `dinov2_vitb14_reg_lc`: **ViT-Base**, patch `14×14`, quattro register tokens e una testa lineare upstream poi sostituita per il task degli ingredienti. La variante ViT-B/14 distribuita da DINOv2 ha circa 86 milioni di parametri; il backbone distilled ha dimensione nascosta `D = 768`, 12 blocchi encoder e 12 attention head per blocco. L'MLP interno è quello standard dei ViT distilled, con espansione tipicamente `4D` (3072 unità nascoste) prima della proiezione di ritorno a 768.

| Componente | Configurazione ViT-B/14 | Conseguenza pratica |
| --- | --- | --- |
| Patch embedding | conv/proiezione lineare su patch `14×14` | trasforma regioni locali in token, senza convoluzioni successive |
| Token visuali a `224×224` | `16×16 = 256` | griglia spaziale a risoluzione 1/14 dell’immagine |
| Dimensione del token | 768 | dimensione di ogni feature propagata dal transformer |
| Encoder | 12 blocchi | contesto globale costruito iterativamente |
| Multi-head attention | 12 head, 64 dimensioni/head | relazioni fra patch in sottospazi diversi |
| Feed-forward network | 768 → 3072 → 768 | trasformazione non lineare token-wise dopo l’attenzione |
| Parametri | ~86 M nel backbone ViT-B | più capiente dei CNN custom del progetto; il costo cresce con il numero di token |

I dettagli `768 / 12 / 12` e l'uso di un MLP per la variante ViT-B distilled sono riportati nella tabella architetturale del lavoro DINOv2. La scelta `B/14` è un compromesso: patch più piccole aumentano la risoluzione delle feature ma quadruplicano circa il costo dell'attenzione quando il numero di token raddoppia per lato.

### Dall'immagine alla sequenza di token

Data un input $x \in \mathbb{R}^{B \times 3 \times H \times W}$ con $H$ e $W$ multipli di 14, il patch embed divide l'immagine in patch non sovrapposte. Il numero di patch è:

$$N = \frac{H}{14}\frac{W}{14}.$$

Ogni patch $p_i \in \mathbb{R}^{3 \cdot 14 \cdot 14}$ viene proiettata in $e_i \in \mathbb{R}^{768}$. Operativamente è equivalente a una `Conv2d(3, 768, kernel_size=14, stride=14)`, seguita da flatten della griglia. A `224×224` il risultato è una sequenza di 256 embedding; a `518×518` è una griglia di `37×37 = 1369` patch token. Le positional embedding interpolabili forniscono al transformer l'informazione di posizione che l'attenzione, da sola, non contiene.

Alla sequenza il modello aggiunge un **class token** appreso, destinato a raccogliere informazione a livello di immagine, e quattro **register tokens** appresi. I register non rappresentano regioni dell'immagine: sono slot di memoria globali senza semantica spaziale prefissata. La sequenza elaborata assume quindi la forma:

$$Z_0 = [c; r_1; \ldots; r_R; e_1; \ldots; e_N] + P,$$

dove $c$ è il class token, $r_j$ sono i quattro registri e $P$ indica le componenti posizionali applicabili ai token visuali. A `224×224` il modello elabora quindi `1 + 4 + 256 = 261` token. I registri hanno un costo di attenzione piccolo a risoluzioni usuali ma non nullo.

### Cosa fa un blocco transformer

Ogni blocco ViT-B segue una struttura **pre-normalization** con due rami residui. Per una sequenza $Z_{l-1}$:

$$U_l = Z_{l-1} + \operatorname{MSA}(\operatorname{LN}(Z_{l-1})),$$
$$Z_l = U_l + \operatorname{MLP}(\operatorname{LN}(U_l)).$$

In DINOv2 sono presenti anche meccanismi di stabilizzazione del training, quali LayerScale e stochastic depth nella ricetta di pretraining; nella variante ViT-B distilled il drop rate dichiarato è zero. La normalizzazione prima di ogni sottoblocco lascia un percorso residuale diretto, favorevole alla propagazione del gradiente in profondità.

La multi-head self-attention calcola, per ogni head, query, key e value:

$$Q = ZW_Q, \qquad K = ZW_K, \qquad V = ZW_V,$$
$$\operatorname{Attn}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.$$

Per ViT-B, $d_h = 768/12 = 64$. Ogni token può assegnare peso a tutti gli altri token — patch lontane, class token e register — e non soltanto a un vicinato locale come in una convoluzione. Le 12 head apprendono matrici $W_Q,W_K,W_V$ distinte e le loro uscite vengono concatenate e proiettate nuovamente a 768 dimensioni. L'attenzione ha complessità quadratica $O(N^2D)$ rispetto al numero di patch: passando da 224 a 448 pixel, $N$ passa da 256 a 1024 e la parte quadratica dell'attenzione cresce di circa 16 volte.

Dopo l'attenzione, l'MLP opera indipendentemente su ciascun token: `Linear(768, 3072) → GELU → Linear(3072, 768)`. L'attenzione mescola informazione tra token; l'MLP aumenta la capacità non lineare della rappresentazione di ogni token dopo tale aggregazione. La ripetizione di questi due meccanismi consente alle patch di costruire feature contestualizzate, non semplici descrittori locali.

### Perché i register tokens

I register sono stati introdotti nel lavoro *Vision Transformers Need Registers*. Gli autori osservano che alcuni token di patch a basso contenuto informativo, spesso sullo sfondo, assumono norme anormalmente alte e vengono riutilizzati dal ViT come spazio di calcolo globale; ciò produce artefatti nelle mappe di attenzione e nelle feature dense. Aggiungere token appresi dedicati a questa funzione separa la memoria di lavoro dall'immagine, rendendo le feature di patch e le attention map più regolari.

Per la classificazione di ingredienti, i register non sono direttamente “ingredient tokens” né sostituiscono il class token. Il loro contributo è indiretto: migliorano la qualità delle feature con cui class token e patch token codificano gli oggetti, le texture e le relazioni contestuali del piatto. Questo è particolarmente rilevante per le applicazioni di interpretabilità: una mappa attentiva più pulita non dimostra causalità, ma riduce un'importante sorgente strutturale di artefatti del backbone.

## Cosa è stato appreso nel pretraining

### Student–teacher senza etichette

DINOv2 è preaddestrato in modo discriminativo self-supervised: non usa etichette di classe per fornire il target. Un **student** riceve crop diverse della stessa immagine e viene ottimizzato; un **teacher** genera target sulle proprie crop ed è aggiornato come media mobile esponenziale (EMA) dei pesi dello student, non mediante backpropagation. In termini schematici:

$$\theta_t \leftarrow m\theta_t + (1-m)\theta_s,$$

dove il momentum $m$ cresce con un cosine schedule durante il training. Il teacher è quindi una versione temporalmente più stabile dello student e rende possibile un obiettivo di distillazione senza label.

### Obiettivo globale DINO e obiettivo locale iBOT

La loss DINO a livello immagine confronta le distribuzioni sui *prototype scores* ottenuti dal class token dello student e del teacher su view differenti della stessa immagine. Dopo softmax e centering del teacher, la forma è una cross-entropy:

$$\mathcal{L}_{\mathrm{DINO}} = -\sum_k p_t(k)\log p_s(k).$$

Questa parte spinge il class token a restare consistente fra crop differenti: è la componente utile a una classificazione globale.

La loss iBOT opera invece sulle patch mascherate: allo student alcune patch vengono nascoste, mentre il teacher osserva la vista non mascherata; si confrontano le distribuzioni delle patch corrispondenti. In forma compatta:

$$\mathcal{L}_{\mathrm{iBOT}} = -\sum_{i \in \mathcal{M}}\sum_k p_{t,i}(k)\log p_{s,i}(k).$$

Il risultato è importante: il backbone non deve solo riassumere l'immagine nel class token, ma deve preservare feature dense a livello di patch. DINOv2 usa head MLP distinte per obiettivo DINO e iBOT, diversamente da alcune ricette precedenti che condividevano la proiezione.

Alla loss si aggiunge il regolarizzatore **KoLeo**, basato sulla distanza dal nearest neighbour tra feature normalizzate nel batch. Massimizzando la dispersione delle feature evita che lo spazio di rappresentazione collassi o concentri eccessivamente molti esempi nella stessa regione. L'obiettivo reale di pretraining è quindi una combinazione pesata di consistenza globale, predizione locale delle patch e dispersione geometrica.

### Scala, curazione e distillazione

Il risultato DINOv2 non deriva soltanto dalla loss: il paper attribuisce un ruolo essenziale a LVD-142M, dataset curato automaticamente di 142 milioni di immagini, alla ricetta di training scalabile e alla distillazione. Il grande ViT-g/14 viene addestrato da zero; i modelli più piccoli, incluso ViT-B/14, vengono distillati dal teacher più grande. Per questo il ViT-B usato qui porta conoscenza trasferita da un modello molto più capiente pur restando gestibile come backbone downstream.

È utile separare questa fase dal training del repository: qui non vengono riapplicate DINO loss, iBOT o EMA. La pipeline corrente usa i pesi risultanti come feature extractor e ottimizza solo la loss multi-label supervisionata sugli ingredienti.

## Relazione con il task degli ingredienti

La variante `_lc` non classifica con il solo class token. Nel suo default hub `layers=4`, concatena i class token normalizzati degli ultimi quattro blocchi e la media dei patch token normalizzati dell'ultimo blocco. La feature che entra nell'head ha pertanto dimensione `5 × 768 = 3840`:

$$h = [c_{9}; c_{10}; c_{11}; c_{12}; \operatorname{mean}(E_{12})].$$

L'head ImageNet upstream è `Linear(3840, 1000)`. Il wrapper del repository ne legge la dimensione in ingresso e la sostituisce con `Linear(3840, num_classes)`, conservando dunque l'aggregazione multi-layer e globale già realizzata dall'implementazione DINOv2. Nel caso multi-label, ciascuna riga della matrice della nuova head corrisponde a un ingrediente e apprende una direzione nello spazio delle feature che ne aumenta il logit. Non esiste softmax fra ingredienti: “pomodoro”, “basilico” e “olio” possono ricevere simultaneamente logit alti.

Il pretraining globale favorisce la robustezza a crop, stile, sfondo e variazioni di dominio; la componente patch-level preserva dettaglio utile a ingredienti piccoli o localizzati. Tuttavia, una foto di piatto non garantisce osservabilità completa della ricetta: alcuni ingredienti possono essere mescolati, nascosti o inferibili solo dal contesto. DINOv2 può apprendere correlazioni visive e semantiche del training set, ma non trasforma un'informazione non visibile in evidenza certa. Soglie di decisione, class imbalance e co-occorrenze restano quindi responsabilità della parte supervisionata della pipeline.

## Costruzione dell'oggetto e grafo effettivo

La classe concreta fissa `weights = "dinov2_vitb14_reg"`; il costruttore base aggiunge il suffisso `_lc` e invoca:

```python
torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg_lc")
```

La variante è un ViT-Base a patch `14×14` con token di registro (*register tokens*). Con il formato di input usuale `224×224`, la patch embedding produce una griglia `16×16`, cioè 256 token visuali, a cui il backbone aggiunge i token speciali. L'attenzione lavora quindi globalmente sulla sequenza di token anziché su un campo ricettivo convoluzionale locale. I token di registro sono parte del backbone upstream e non sono gestiti individualmente dal codice del progetto.

Subito dopo il caricamento, il repository sostituisce la testa upstream:

```python
self.model.linear_head = nn.Linear(
    self.model.linear_head.weight.shape[1], num_classes
)
```

Il numero di feature in ingresso viene letto dall'head caricata, non hard-coded. Questo rende l'adattamento resistente a cambiamenti della dimensione di embedding del modello hub selezionato, ma presuppone che il modello esponga proprio l'attributo `linear_head`.

Il grafo usato dal `forward` è quindi:

```text
immagine normalizzata
  → patch embedding e transformer DINOv2
  → rappresentazione globale prodotta dal wrapper upstream
  → nuova linear_head [feature_dim → num_classes]
  → logit per ingrediente
```

La classe delega interamente il forward a `self.model(x)`: non estrae patch, token CLS o register token in modo esplicito. Cambiamenti nel contratto del wrapper upstream di DINOv2 possono quindi influire direttamente sull'integrazione.

## Pretraining, congelamento e modalità di fine-tuning

### Cosa controlla davvero `pretrained`

`DinoV2B14` accetta e serializza il parametro `pretrained`, ma il valore non condiziona la chiamata a `torch.hub.load`. Il codice carica sempre il modello hub identificato da `dinov2_vitb14_reg_lc`; `pretrained=False` non costruisce un ViT a pesi casuali. Va quindi interpretato come metadato di configurazione, non come un interruttore funzionale.

### Linear probing (default)

Con `freeze_backbone=True` — anche `None` viene convertito a `True` — `freeze_backbone()` visita `self.model.backbone.named_parameters()` e imposta `requires_grad=False`. La nuova `linear_head`, che è esterna a `model.backbone`, resta allenabile. Questa è una configurazione di *linear probing*: durante la backward pass i gradienti non aggiornano il backbone, ma l'ottimizzatore viene comunque costruito con `self.model.parameters()` e include anche i parametri congelati; PyTorch li ignora perché non ricevono gradiente.

È una scelta appropriata quando il dataset è piccolo o si vuole isolare la qualità delle feature DINOv2, ma riduce la capacità di adattare rappresentazioni generiche alla semantica degli ingredienti, spesso visivamente sottili o parzialmente occlusi.

### Fine-tuning completo

`unfreeze_backbone()` riabilita `requires_grad=True` sui parametri del backbone. Non esiste una callback o una policy automatica che lo invochi dopo N epoche: il passaggio da linear probing a fine-tuning deve essere orchestrato esplicitamente dalla configurazione/codice di training.

Il metodo non ricostruisce l'ottimizzatore. Se viene chiamato dopo che Lightning ha già creato l'ottimizzatore, i parametri sono già presenti nei suoi param group e possono iniziare ad aggiornarsi; resta però responsabilità dell'esperimento scegliere learning rate, weight decay e scheduler adatti al cambiamento di regime.

### Layer-wise pretraining non supportato

Il costruttore conserva `lp_phase`, ma DINOv2 non implementa `_lp_init_layers`, `_lp_step_phase` o gli altri hook del protocollo LP di `BaseModel`. L'argomento deve restare a `-1`: un valore attivo non costruisce una versione progressiva del transformer e l'uso di `lp_phase_step()` porterebbe al comportamento base non implementato. Non va confuso con il congelamento del backbone.

## Preprocessing: pipeline dichiarata e pipeline realmente eseguita

### Trasformazioni dichiarate dal modello

Senza override, la proprietà `transform_aug` restituisce `transform_aug_dino`:

1. `RandomResizedCrop(input_shape)` con `scale=(0.2, 1.0)`, `ratio=(0.75, 1.3333)` e interpolazione bicubica;
2. flip orizzontale con probabilità 0.5;
3. conversione a `float32` e scaling in `[0, 1]`;
4. normalizzazione con media `[0.485, 0.456, 0.406]` e deviazione `[0.229, 0.224, 0.225]`.

`transform_plain` restituisce invece resize del lato corto a 256 (mantenendo il rapporto), center crop a `input_shape`, conversione/scaling e la stessa normalizzazione. Per il valore predefinito `input_shape=224`, la risoluzione è compatibile con il patch size 14 (`224 / 14 = 16`).

Con `trns_aug` custom il modello usa `transform_core_dino`: train usa `RandomResizedCrop`, validazione usa `Resize(256) → CenterCrop`; poi inserisce le augmentation custom prima della conversione e della normalizzazione DINO. Se invece si passano `trns_bld_aug` e/o `trns_bld_plain`, essi sostituiscono direttamente i builder di default.

### Normalizzazione aggiunta dal DataModule: stato corrente

È importante distinguere il builder del modello dalla trasformazione consumata dal dataset. In `BaseDataModule.prepare_data()`, `_init_transform()` tratta una lista di trasformazioni come input per `transformations_wrapper()`, che aggiunge sempre:

```text
ToImage → trasformazioni ricevute → ToDtype(float32, scale=True)
        → Normalize(mean=train_images_stats, std=train_images_stats)
```

Poiché i builder DINO di default restituiscono una **lista** che contiene già `ToDtype` e `Normalize(DINO_MEAN, DINO_STD)`, la pipeline effettiva applica due normalizzazioni consecutive: prima le statistiche DINO, poi le statistiche calcolate sul training set. Lo stesso succede sia in train sia in validation/test/predict.

Questa composizione non coincide con il preprocessing DINO standard e altera la distribuzione attesa dal backbone preaddestrato. È un comportamento dell'implementazione corrente, non una raccomandazione per nuove run. Un builder custom che restituisce direttamente un `v2.Transform` (per esempio un `v2.Compose`) evita il wrapping del DataModule e può quindi definire in modo esplicito una sola normalizzazione. In ogni caso `images_stats_path` resta richiesto da `prepare_data()` anche quando il transform già gestisce la normalizzazione.

## Batch size, accumulo e memoria

`_BaseDinoV2.MAX_ALLOWED_BATCH_SIZE = 32`. `BaseLGNM` usa questo limite per calcolare batch reale e accumulo dei gradienti prima di costruire DataModule e Trainer:

```text
target_batch_size ≤ 32  → real batch = target, accumulo = 1
target_batch_size > 32  → accumulo = ceil(target / 32)
                         real batch = ceil(target / accumulo)
```

Per esempio, una configurazione con batch target 128 viene eseguita come micro-batch 32 e `accumulate_grad_batches=4`. Per valori non divisibili, il prodotto `real_batch × accumulo` può essere leggermente maggiore del target; il commento nel codice indica una divisibilità esatta, ma l'implementazione usa arrotondamenti verso l'alto.

Il limite è una convenzione applicativa, non una garanzia di assenza di OOM: dipende da risoluzione, precisione, stato dell'ottimizzatore, ampiezza dell'head, numero di worker e GPU. I trainer veloci/Optuna impostano precisione `16-mixed`, che può ridurre l'uso di memoria, mentre il trainer base usa la propria configurazione di precisione.

## Checkpoint, configurazione e ripresa

`to_config()` aggiunge a `BaseModel.to_config()` i flag `pretrained` e `freeze_backbone`. La ricostruzione usa `load_from_config()`, quindi esegue nuovamente `torch.hub.load` prima di caricare lo state dict del checkpoint. Per riprendere una run serve pertanto che il repository hub sia raggiungibile oppure già presente nella cache locale di torch.hub; il checkpoint del progetto non elimina questa dipendenza in fase di costruzione del modello.

La configurazione salva anche callable di trasformazione quando vengono passate come override. Queste sono oggetti Python del processo, non una definizione JSON portabile: una ripresa affidabile richiede che le funzioni e gli import corrispondenti restino disponibili.

## Interpretabilità e limiti degli hook

`classifier_target_layer` restituisce `self.model.linear_head`, quindi rappresenta correttamente l'ultimo mapping verso gli ingredienti. Per Grad-CAM e factorization, `conv_target_layer` restituisce `self.model.backbone.blocks[-1].norm1`: è l'ultima normalizzazione pre-attention, le cui attivazioni influenzano direttamente l'ultimo blocco e l'head. DINOv2 non ha un ultimo layer convoluzionale; `gradcam_reshape_transform` rimuove CLS e i quattro register token, quindi converte i rimanenti patch token da `[B, 256, 768]` a `[B, 768, 16, 16]` per l'input standard `224×224`.

Questo reshape è indispensabile perché Grad-CAM e Deep Feature Factorization lavorano su mappe spaziali `[B, C, H, W]`, mentre il ViT produce sequenze. Il wrapper Grad-CAM rende inoltre l'input differenziabile: senza questo passaggio, un backbone congelato non produce gradienti al target layer e Grad-CAM non può calcolare i pesi della mappa.

Quando si interpretano predizioni per ingrediente, la heatmap risultante dipende anche da come l'utilità di visualizzazione riconverte token e feature; la sola scelta di `backbone.norm` non garantisce una mappa spaziale semanticamente equivalente a Grad-CAM su una CNN.

## Checklist per una nuova run

- Usare `DinoV2B14` e fornire il numero di classi dall'encoder della run, mai un numero fissato manualmente.
- Verificare disponibilità/caching di `facebookresearch/dinov2` nell'ambiente di esecuzione prima di un job lungo.
- Scegliere consapevolmente fra linear probing (`freeze_backbone=True`) e fine-tuning (`False` oppure `unfreeze_backbone()` orchestrato esplicitamente).
- Tenere `lp_phase=-1` per DINOv2.
- Controllare la pipeline finale del DataModule: con i builder di default oggi è presente la doppia normalizzazione descritta sopra.
- Per input `224×224`, mantenere crop e resize multipli di 14; dimensioni diverse richiedono la verifica del contratto del modello hub e del numero di token.
- Trattare `max_allowed_batch_size=32` come punto di partenza e verificare la memoria sul device e sulla precisione effettivamente scelti.

## Riferimenti nel repository

- `src/models/dinov2.py`: wrapper, head, congelamento, hook di visualizzazione e limite batch.
- `src/data_processing/transformations.py`: builder `transform_*_dino` e statistiche di normalizzazione.
- `src/data_processing/common.py`: wrapping delle liste di trasformazioni e dipendenza da `train_images_stats.csv`.
- `src/lightning/lgn_models.py`: batch effettivo, accumulo e costruzione dell'ottimizzatore.
- `src/training/commons.py`: propagazione di trasformazioni e batch dal Lightning model al DataModule.

## Fonti primarie

- Oquab et al., [*DINOv2: Learning Robust Visual Features without Supervision*](https://arxiv.org/abs/2304.07193): architettura ViT-B, pretraining DINO+iBOT, KoLeo, LVD-142M e distillazione.
- Darcet et al., [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588): motivazione e funzionamento dei register tokens.
- [Repository ufficiale `facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2): nomi dei modelli pubblicati e caricamento con PyTorch Hub; l'[implementazione della testa lineare](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/classifiers.py) definisce i quattro register e l'aggregazione degli ultimi quattro blocchi usata da `_lc`.
