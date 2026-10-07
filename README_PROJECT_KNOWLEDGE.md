# Ingredient Recognition — conoscenza del progetto

> Documento vivente per l'assistente e per chi lavora al repository. Va aggiornato a ogni modifica architetturale o funzionale rilevante, e quando si confermano nuove informazioni sul progetto.

**Ultimo aggiornamento:** 7 ottobre 2026
**Stato della ricognizione:** architettura e flusso principale verificati nel codice. `ingredients_target_v5_metadata.json` è il default runtime FoodOn-first, con 165 target e split Yummly 47.965/5.996/5.996 train/val/test; `v4` e le generazioni legacy restano disponibili. La compatibilità storica 2.1c e Data 2.4 sono chiuse: training CUDA minimo, checkpoint reload e dashboard sono verificati nel [contratto runtime](docs/implementation_details/image_data_loading.md). Il selettore 4B-D1 EfficientNetV2-S è implementato. La campagna Phase 3-D1/D2/D3 `phase3-d1-v3` ha completato 40 epoche con batch effettivo 128; D4 e la sua applicazione originale restano conservati. D5 rende facoltative le revisioni manuali. D6 adotta la politica di qualità validation con intervalli appaiati coerenti, mantenendo train e cuisine come diagnostiche. La proiezione condivisa `ingredients_selected_v5_d6_v1` è un artefatto esplicito e versionato, integrato in P7 come opzione a 59 label e non come nuovo default runtime. P7 chiude la macrofase 3 con parità e retention verificate; lo stato corrente è in `docs/general_plan.md`. Il confronto esplorativo storico `basic_v5` è conservato sotto `docs/experiment_results/` senza modificare il gate del benchmark finale.

## Scopo

Progetto di tesi per predire gli ingredienti di una ricetta a partire dalla sua immagine. Il problema è formulato principalmente come classificazione multi-label: per una foto il modello produce un logit/probabilità per ogni ingrediente del vocabolario.

Il repository contiene anche esperimenti esplorativi su rappresentazioni testuali delle ricette e sui flavour, ma il percorso attivo e maggiormente strutturato è quello di visione artificiale con immagini e ingredienti.

## Flusso principale

```text
Dataset raw (Yummly / Recipes1M / recipes)
  -> script raw2input: unione, riordino, download/copia immagini, split train/val/test
  -> data/input/<dataset>/{train,val,test}/
       immagini + metadata.json
  -> ImagesRecipesBaseDataModule
       filtro per cucina, codifica ingredienti, pesi di classe, trasformazioni
  -> Lightning model
       backbone (ResNet, DenseNet, DINOv2, oppure modello custom)
       + BCEWithLogitsLoss e metriche multi-label
  -> trainer Lightning
       checkpoint, CSV/TensorBoard/W&B, eventuale early stopping
  -> experiments/
       trial e configurazioni serializzate
  -> dashboard Dash / Optuna / TensorBoard
```

## Dati

### Percorsi e convenzioni

`settings/config.py` centralizza le configurazioni e le costanti del progetto, inclusi percorsi assoluti calcolati a partire dalla root del repository, parametri predefiniti e impostazioni operative come W&B. I dati elaborati vivono in `data/input`; quelli sorgente in `data/raw_input`.

Il `DataModule` di default usa `data/input/yummly`. I metadata restano in `train/`, `val/` e `test/`, mentre tutte le immagini sono risolte da `imgs/standard/`; quando non esiste uno split `predict`, viene riutilizzato `test`. `ingredients_target_v4_metadata.json` è un baseline validato, mentre `ingredients_target_v5_metadata.json` è il nuovo default runtime. `metadata.json` e `sel_ing_2410_metadata.json` con `ingredients_ok` restano generazioni legacy immutabili.

Per ogni ricetta il codice si aspetta almeno il campo selezionato da `feature_label`, un'immagine nel campo `image` e, se si filtra, la cucina nel campo `cuisine`. Il default corrente per le nuove configurazioni è `ingredients_target`, derivato da `ingredients`, mentre gli esperimenti storici mantengono esplicitamente `ingredients_ok`. Il filtro ammette: american, chinese, french, greek, indian, italian, japanese, mexican, spanish, thai e all.

### Preparazione

- `src/raw2input/yummly/recipes_merge.py` riunisce le ricette raw di Yummly in `all_recipes.json`.
- `src/raw2input/yummly/sort_recipes_as_img.py` ordina le ricette in base alle immagini disponibili.
- `src/raw2input/yummly/creation.py` crea gli split (seed 42; val/test 8% ciascuno), copia le immagini disponibili e genera i relativi metadata.
- `src/raw2input/recipes1M/extraction.py` è uno script parzialmente operativo per campionare ricette, scaricarne immagini e preparare Recipes1M; contiene ancora porzioni commentate.
- `src/raw2input/compute_img_stats.py` calcola media e deviazione standard RGB dello split di training e salva `train_images_stats.csv`. Il file è richiesto dal DataModule base per le trasformazioni standard.

### Etichette e split

`ImagesRecipesBaseDataModule` carica tutti gli split, applica il filtro per cucina e adatta/usa un encoder multi-label. Per il nuovo `ingredients_target` usa il `MultiLabelBinarizer` stretto: il vocabolario viene appreso dal train, salvato nella configurazione e produce 165 output senza `<UNK>`. Un'etichetta estranea al vocabolario causa un errore esplicito. I field legacy continuano a usare `MultiLabelBinarizerRobust`, e le configurazioni storiche ricostruiscono il loro `<UNK>` e la dimensione di output salvata. I pesi di classe sono calcolati dalle frequenze dello split train e possono essere usati dalla loss.

I quattro DataLoader immagini condividono una policy `pin_memory` portabile. Il valore predefinito `None` abilita la pinned memory soltanto su Windows nativo e la disabilita su WSL, Linux, macOS e piattaforme non riconosciute; `True` e `False` restano override espliciti. La configurazione serializza la policy non risolta, così lo stesso esperimento si adatta al sistema operativo quando viene ricaricato. `num_workers`, worker persistenti e prefetch restano impostazioni indipendenti. Il contratto completo è in `docs/implementation_details/image_data_loading.md`.

Sono presenti dataset/encoder ulteriori per one-vs-all, classificazione multi-classe, sequenze di ingredienti con token speciali, masking e flavour: sono secondari rispetto alla pipeline immagini → ingredienti.

## Modelli e preprocessing

I modelli di visione integrati nel training discendono da `BaseModel`, che centralizza configurazione, serializzazione e definizione delle trasformazioni train/validation. `DICANetSExperiment` integra il nucleo tensoriale DICA-Net-S in questa interfaccia e nel percorso sperimentale opt-in.

- `src/models/resnet.py`: ResNet custom simili a ResNet-18/50 e wrapper torchvision per ResNet18 e ResNet50, con teste adattate al numero di ingredienti.
- `src/models/densenet.py`: DenseNet custom e wrapper torchvision DenseNet121/DenseNet201; i wrapper torchvision hanno attualmente un difetto verificato nel contratto delle trasformazioni (`self.tr_weights` non inizializzato) e non sono un percorso di training mantenuto finché non viene corretto e sottoposto a smoke test.
- `src/models/dinov2.py`: DINOv2 ViT-B/14 con head lineare sostituita; usa `torch.hub` per caricare `facebookresearch/dinov2` e può congelare il backbone (default).
- `src/models/dummy.py`: modelli minimi per test.

`src/models/efficientnet.py` implementa il wrapper EfficientNetV2-S mantenuto
per il selettore 4B-D1, con head indipendente da 165 logit e full fine-tuning.
Il contratto sperimentale 4A resta distinto dal protocollo selettore.

Le fondamenta opt-in 4A sono in `src/models/experimental_contract.py`,
`src/data_processing/experimental_transforms.py` e `src/training/batching.py`:
identità primitive/versionate, RGB full-frame fit/pad 224, normalizzazione
ImageNet, inizializzazione FP32 delle nuove head, costruzione offline esplicita
e batch effettivo esatto con pesatura dei gruppi finali. Il contratto e i limiti
verificati sono in [`docs/implementation_details/experimental_model_contract.md`](docs/implementation_details/experimental_model_contract.md).
`EfficientNetV2SExperiment` in `src/models/experimental_efficientnet.py` è ora
l'adapter 4A distinto a 224, con feature intatte, nuova head pooled senza dropout
e modalità full/frozen con encoder persistente in eval. `ExperimentalLGNM` e
`src/training/experimental_runtime.py` integrano accumulo esatto, ordine delle
classi, mapping rigoroso dell'encoder e ripristino completo offline con identità
conservate nei checkpoint full/light. Training, dashboard e ricostruzione del
trial migliore usano questo percorso opt-in; il default legacy, il selettore e
il vocabolario completo restano invariati. `MaxViTTExperiment` in
`src/models/experimental_maxvit.py` conserva stem e blocchi MaxViT-T e usa la
stessa head GAP/lineare a 224. La policy BatchNorm epsilon 1e-3/momentum 0.01,
la geometria delle partizioni e gli indici relativi sono verificati e salvati;
le statistiche preaddestrate restano intatte. Pesi ufficiali, hash completo,
ripristino offline e helper Grad-CAM/factorization su input sintetici sono
verificati. La qualificazione CUDA/consumer reali resta da fare.

`src/models/dica_net.py` implementa `DICANetSCore` e `DICAReadoutS`: **DICA-Net**
significa *Dual-scale Ingredient-query and Context Attention Network* ed è il
nome adottato per la proposta storica P2, alla scala S. Il nucleo esegue una
sola volta le feature EfficientNetV2-S fornite dal chiamante, usa i tap 5/7 per
245 token e somma i logit del ramo query con quelli del contesto globale.
Flusso, gradienti, attenzione numerica e invarianti delle righe delle label sono
verificati su CPU. `src/models/experimental_dica_net.py` implementa l'adapter
`DICANetSExperiment`: inizializzazione dall'artefatto ImageNet originale con
hash completo, encoder intatto, modalità full/frozen e persistenza canonica
full/light offline. La configurazione conserva topologia `p2_s`, provenienza e
BatchNorm nativa epsilon 1e-3/momentum 0.1; ordine delle classi e proiezione
rimangono verificati dal runtime comune. La testa completa richiede entrambe
le scale: la factorization dei singoli concetti è esplicitamente non supportata,
non sostituita con il solo ramo globale. L'integrazione diagnostica dei consumer
e la qualificazione GPU restano aperte. Il [contratto DICA-Net](docs/implementation_details/experimental_model_contract.md#dica-net-s-initialization-and-persistence--542)
documenta API, artefatto e limiti; questi controlli non avviano una campagna.

Le direttive complete sulla collocazione delle informazioni, sulle fonti autorevoli, sul ciclo di vita e sulla conservazione a lungo termine sono in `docs/README_DOCS_ORGN.md`. Deve essere letto insieme a `docs/README.md` prima di creare, spostare o modificare sostanzialmente un documento.

La struttura, la lingua e la metodologia di scrittura della documentazione sono definite in `docs/README.md`. Tutti i documenti sotto `docs/` devono essere scritti in inglese. Per approfondimenti tecnici sulle architetture e sulla ricerca di riferimento, consultare `docs/models_deepdive/`. Al momento è disponibile `docs/models_deepdive/dinov2.md`, dedicato a DINOv2 ViT-B/14; la panoramica dei modelli è in `docs/implementation_details/models.md`. Il contratto permanente e autorevole dei mapping custom da `ingredients` a `ingredients_target`, incluse esclusioni, espansioni multi-target e collisioni vietate, è `docs/implementation_details/ingredient_mapping_rules.md`; deve essere aggiornato insieme allo standardizzatore e ai relativi test.

L'obiettivo di ricerca e i relativi audit dei dati sono formalizzati in `docs/project_objective/`. Il dataset attivo è esclusivamente Yummly: `yummly_data_audit.md` documenta il dataset e i limiti delle etichette legacy, mentre `ingredient_vocabulary_audit.md` analizza il candidato da 209 target. L'esperimento Yummly 2.2c e il relativo gate decisionale sono mantenuti insieme al piano attivo in `docs/plans/data_ingredient_refactor/controlled_vocabulary_evaluation.md`. Il catalogo generale e riusabile dei vocabolari è invece in `docs/research/topics/ingredient_vocabularies/`. Le decisioni vincolanti sono in `docs/project_objective/benchmark_decisions.md`: mantenere `feature_label` configurabile con nuovo default `ingredients_target`, rigenerare quel campo da `ingredients` con regole deterministiche, scegliere la granularità fine soprattutto in base alla riconoscibilità nel piatto preparato, usare solo controlli automatici sulle immagini, raggruppare soltanto immagini byte-identiche tramite SHA-256, creare uno split 80/10/10 bilanciato, evitare manifest duplicati, preservare senza riscriverlo il set minimo selezionato di evidenze legacy e i suoi checkpoint ancora eseguibili, rimuovere `<UNK>` dai nuovi output multi-label mantenendolo negli artefatti legacy selezionati, usare mAP macro e micro F1 come metriche primarie abbinate e calibrare/sogliare soltanto sulla validation.

Le evidenze sui candidati EfficientNetV2, Swin V2, SigLIP2, testa query/set e MaxViT sono raccolte in `docs/research/topics/experimental_model_candidates/`; brief, componenti, compatibilità e tre proposte custom sono in `docs/research/topics/custom_attention_model_design/`. La scelta vincolante è in [`docs/project_objective/experimental_model_portfolio.md`](docs/project_objective/experimental_model_portfolio.md): 4A-D1 adotta EfficientNetV2-S e MaxViT-T; 4A-D2 adotta DICA-Net-S (proposta storica P2-S), con encoder EfficientNetV2-S intatto, query per ingrediente su due scale e ramo globale. Il fallback mantiene S ma congela l'encoder, cambiando il protocollo di adattamento. Tutti e tre gli adapter 4A sono implementati e testati a livello d'interfaccia/persistenza; l'integrazione diagnostica DICA-Net e la qualificazione CUDA/consumer reali restano aperte. Fanno fede codice e `docs/implementation_details/models.md` per le funzionalità disponibili. La scelta indipendente del selettore è conclusa e implementata: [`docs/project_objective/model_comparison_methodology.md`](docs/project_objective/model_comparison_methodology.md) congela 4B-D1 con pesi `EfficientNet_V2_S_Weights.IMAGENET1K_V1`, full fine-tuning, preprocessing RGB full-frame a 384 px, head pooled indipendente, BCE pesata dal solo train e target operativo FP32/batch fisico 8; [`docs/implementation_details/ingredient_selection.md`](docs/implementation_details/ingredient_selection.md) descrive il relativo runtime mantenuto. Questa coincidenza di famiglia con 4A non trasferisce ranking o risultati tra i due ruoli.

Lo stato di avanzamento dell'intero progetto di tesi è mantenuto in `docs/general_plan.md`. Il tracker conserva lo storico ed è organizzato nelle macro-sezioni fondazione, dati, selezione degli ingredienti, ricerca e implementazione di modelli aggiuntivi, training e hyperparameter tuning, confronto dei risultati e scrittura della tesi. I piani esecutivi delle implementazioni concrete sono mantenuti separatamente in `docs/plans/`. L'audit del vocabolario candidato (2.2a) è completato e l'intero perimetro 2.2b è approvato: `chili powder` e le forme generiche powdered, ground, crushed, dried-crushed e flaked red pepper confluiscono in `chili`, mentre i peperoni rossi e verdi freschi restano distinti. La compatibilità legacy 2.1c è chiusa sul solo set minimo di evidenze della selezione ResNet 2024; il manifest e gli smoke test restano read-only e una eventuale pulizia richiede una decisione separata.

Il piano esecutivo attivo della fase Data è `docs/plans/data_ingredient_refactor/yummly_data_phase.md`. Copre i Work package 2.1b–2.4: store immagini condiviso, compatibilità degli esperimenti storici, standardizzatore `ingredients_target`, audit del vocabolario (2.2a), rafforzamento dell'estrattore (2.2b), ricerca del vocabolario controllato (2.2c), implementazione della nuova pipeline (2.2d), split deterministico con gruppi SHA-256 esatti e integrazione runtime senza `<UNK>` nei nuovi output multi-label. La compatibilità legacy (2.1c) è chiusa sul set minimo H1–H4 e sui checkpoint anchor esplicitamente elencati nel piano; il verificatore read-only è `scripts/validate_legacy_experiments.py` e il manifest è conservato in `src_scratches/ingredient_selection_reconstruction/retention_manifest.json`. `v4` e `v5` sono conservati come generazioni riproducibili e `v5` è il default runtime. La direzione 2.2c usa FoodOn come autorità primaria: dopo la pulizia meccanica tenta il match FoodOn, usa le regole locali solo come fallback e riprova il match esatto; match assenti o ambigui restano concetti locali. Il fuzzy matching è vietato nella pipeline standard dopo una valutazione empirica che ha rilevato collisioni semantiche anche con un recupero tipografico molto limitato. La nuova selezione è governata da `docs/plans/recognizable_ingredient_selection.md`: conserva la regola storica basata sul massimo train F1 soltanto come baseline e rifà lo studio sul vocabolario `v5` con dinamiche di apprendimento, stabilità, controlli e osservabilità separati. La risalita automatica della gerarchia resta vietata. Mapping espliciti e revisionati verso concetti genitore restano una possibilità deferred per esperimenti separati di riduzione della difficoltà. `ingredients_target` è l'unico vocabolario standard condiviso da split e modelli; eventuali subset devono essere esperimenti nominati e non un secondo default. Lo script `src_scratches/data_anlysis/ingredient_threshold_sweep.py` e i report in `src_scratches/data_anlysis/outputs/controlled_target_generation/` calcolano sul solo train dimensione del vocabolario, copertura, ricette con 0/1/2/3+ target e ingredienti persi a ogni soglia. Per audit puntuali di un metadata, split e campo usare `src_scratches/data_anlysis/metadata_field_audit.py`, che produce conteggi completi e distribuzioni per ricetta senza modificare i dati. Prima di modificare loader, layout Yummly, metadata o builder occorre leggere sia il piano generale sia questo piano esecutivo e mantenerne sincronizzati i tracker.

Aggiornamento 2.4 del 6 agosto 2026: il riferimento precedente a `v5` come candidato non integrato è superato. `v5` è ora il default runtime; l'implementazione senza `<UNK>` per i nuovi target è completa e rimangono soltanto gli smoke test ML dipendenti dall'ambiente.

Aggiornamento 2.4 del 6 ottobre 2026: anche gli smoke reali sono completati. Lo script [`scripts/validation/data_runtime_smoke.py`](scripts/validation/data_runtime_smoke.py) esegue quattro aggiornamenti CUDA, ricarica il checkpoint e verifica la dashboard con cache e output isolati. Il vecchio run incompleto resta conservato. La preparazione canonica legge i metadata di tutti gli split; lo smoke non esegue inferenza o metriche sulle immagini test. Evidenze e limiti sono in [`docs/implementation_details/image_data_loading.md`](docs/implementation_details/image_data_loading.md).

La metodologia del benchmark usa uno split Yummly unico e congelato, stratificato sugli ingredienti e sulle cucine e vincolato dai soli gruppi di immagini SHA-256 identiche; non usa uno split randomico puro. La ricerca generale e le fonti sono in `docs/research/topics/dataset_splitting/split_strategy.md`, mentre contratto, garanzie e limiti dell'implementazione Yummly sono in `docs/technical_details/data/yummly_benchmark_split/explaination.md`; ogni confronto standard tra modelli deve usare gli stessi metadata `v5` e mantenere il test fuori dalle decisioni di selezione.

Il piano operativo per implementare i tre protocolli sperimentali adottati in 4A è [`docs/plans/additional_model_implementation.md`](docs/plans/additional_model_implementation.md). Separa contratto comune, EfficientNetV2-S, MaxViT-T, DICA-Net-S e qualificazione tecnica misurata. Conserva distinto il wrapper del selettore e riusa la pipeline canonica; non avvia tuning, controlli randomici o valutazione test. Le fondamenta, i tre adapter e Lightning sperimentale sono verificati; diagnostica custom e qualificazione CUDA/consumer reali restano aperte. Per il runtime fanno fede codice e contratti di implementazione. La policy di smoke dichiarata non fissa gli hyperparametri del benchmark né un cap GPU misurato.

## Tracker obbligatorio dello stato di avanzamento

`docs/general_plan.md` è la fonte autorevole per lo stato, le priorità, le dipendenze e lo storico operativo del progetto. Deve essere letto integralmente prima di iniziare un'attività progettuale, così da identificare la macro-sezione e il work package pertinenti, rispettarne i gate e non ripetere lavoro già completato o superato. Quando un work package entra nella fase di implementazione concreta, il relativo piano dettagliato deve essere creato o aggiornato in `docs/plans/` e collegato al piano generale. Ogni piano d'implementazione deve contenere il proprio progress tracker e diventa la fonte operativa durante lo sviluppo della feature: viene aggiornato al completamento di ciascuno step, registrandone risultato ed evidenze insieme alle decisioni emerse, al nuovo stato e alla prossima azione. Non è richiesto aggiornarlo durante l'avanzamento intermedio dello step. Il piano generale viene sincronizzato quando il piano della feature è completato, oppure prima soltanto se cambia uno stato, una priorità, una dipendenza, lo scope, un completion gate o un blocco materiale a livello di progetto.

Il tracker generale deve essere aggiornato nella stessa modifica che determina uno dei seguenti eventi a livello di progetto:

- completamento di un piano di feature oppure inizio, rinvio, blocco, riapertura o superamento di un work package del piano generale;
- modifica della priorità, della dipendenza, del completion gate o della prossima azione;
- produzione di un nuovo artefatto permanente, risultato sperimentale o evidenza che cambia lo stato del progetto;
- introduzione di un nuovo work package o di una nuova fase necessaria alla tesi.

Durante l'implementazione ordinaria si aggiorna invece il progress tracker del piano di feature interessato. Quando si aggiorna il tracker generale, occorre mantenere sincronizzati il riepilogo generale, lo stato della macro-sezione, la tabella dei work package, le checklist, la prossima azione e la data di ultima modifica. Ogni transizione significativa deve essere aggiunta al registro storico append-only. Le attività completate o superate non devono essere eliminate: rimangono come storico e vengono marcate rispettivamente `Done` o `Superseded` con il collegamento alla relativa evidenza.

Questo README descrive la conoscenza stabile del repository, ma non sostituisce `docs/general_plan.md` per stabilire cosa sia attualmente in corso o quale attività debba essere eseguita successivamente. Analogamente, `docs/plans/` dettaglia l'esecuzione delle singole implementazioni ma non sostituisce il piano generale come fonte dello stato complessivo.

Il dataset corrente da 65.146 record e 182 etichette resta il riferimento immutabile degli esperimenti storici. I suoi `metadata.json` e `sel_ing_2410_metadata.json`, le configurazioni e i checkpoint non devono essere riscritti. Per nuovi claim comparativi si useranno nuove generazioni `ingredients_target` dopo il superamento dei relativi gate.

Le immagini sono normalmente ridimensionate a 224×224. Per i modelli generici il DataModule applica resize, `TrivialAugmentWide` in training e normalizzazione con statistiche del dataset. I wrapper torchvision usano le trasformazioni/normalizzazioni dei pesi ImageNet. DINOv2 usa normalizzazione ImageNet e crop dedicati. Il protocollo 4B-D1 è un'eccezione esplicita e implementata nel percorso specializzato di Phase 3: usa un canvas 384×384 con fit/pad full-frame, normalizzazione ImageNet e solo flip orizzontale nel training primario.

## Addestramento e valutazione

Le revisioni operative Phase 3-D2/D3 del 27 settembre 2026 richiedono batch
effettivo 128 e 40 epoche nel workspace WSL principale (2 di warm-up e 38 di
cosine decay, audit ogni 2 epoche). Il launcher rilanciabile è
`scripts/launch_exps/ingredient_selection/train_selector.py`: effettua il gate
CUDA su un'intera epoca prima di costruire da zero il modello della campagna.
Le prove 128/64/32/16 hanno prodotto OOM; `EfficientNetV2SSelector` espone quindi
`MAX_ALLOWED_BATCH_SIZE = 8`, con accumulo Lightning 16. Il limite vale per
full fine-tuning FP32 a 384 px sulla RTX 4060. Provenienza, batch fisico ed
effettivo, accumulo, limite VRAM e snapshot dei sorgenti sono nel manifest; il
repository può avere modifiche locali, purché il gate corrisponda esattamente
agli stessi sorgenti. La run v1 interrotta è conservata; il gate incompleto v2
è stato interrotto prima dell'avvio della campagna, e la nuova è identificata
come `phase3-d1-v3`. Il contratto è in
[`docs/implementation_details/ingredient_selection.md`](docs/implementation_details/ingredient_selection.md)
e la decisione vincolante in
[`Phase 3-D3`](docs/project_objective/model_comparison_methodology.md#phase-3-d3--forty-epoch-campaign-amendment), con le regole batch/provenienza di D2.
La campagna v3 è terminata dopo 40 epoche. `scripts/ingredient_selection/analyze_campaign.py`
ha esposto soltanto le 24 label del pilot prima del congelamento della regola;
`scripts/ingredient_selection/report_pilot.py` riproduce le decisioni sul solo
pilot archiviato. I gate numerici, legati agli hash di campagna, pilot, evidenza
e classificatore, sono in
[`Phase 3-D4`](docs/project_objective/model_comparison_methodology.md#phase-3-d4--pilot-frozen-numerical-profile-rule).
I risultati provvisori sono in
[`docs/experiment_results/phase3_d1_v3_pilot.md`](docs/experiment_results/phase3_d1_v3_pilot.md).
P4 ha applicato la stessa regola alle 165 label tramite
`scripts/ingredient_selection/analyze_campaign.py`; il nuovo
`scripts/ingredient_selection/report_campaign.py` verifica la concordanza col
pilot e produce gruppi provvisori nominativi e grafici riproducibili. Il
risultato revisionato è in
[`docs/experiment_results/phase3_d1_v3_full_profile.md`](docs/experiment_results/phase3_d1_v3_full_profile.md).
L'appendice facoltativa conserva il precedente pilot P5 di osservabilità e usa
`scripts/ingredient_selection/observability_review.py` per creare schede cieche
di annotazione e verificare, solo dopo due risposte umane indipendenti,
l'accordo tra revisori. Non è un gate della selezione e non può modificare il
vocabolario, il tuning o il confronto primario tra modelli. La decisione è in
[`Phase 3-D5`](docs/project_objective/model_comparison_methodology.md#phase-3-d5--numerical-selection-and-optional-interpretation-appendix).
La rubrica e i limiti sono in
[`docs/project_objective/ingredient_observability_protocol.md`](docs/project_objective/ingredient_observability_protocol.md).

La revisione D6 usa `scripts/ingredient_selection/review_inclusion.py` e i
moduli `inclusion.py`, `inclusion_statistics.py` e `inclusion_reporting.py`
sotto `src/ingredient_selection/`. Rianalizza gli score salvati, senza modello
o inferenza, con bootstrap sui gruppi di immagini validation byte-identiche:
mediana delle cinque AP finali e differenza appaiata rispetto alla prevalenza.
Regola, sorgenti, report e grafico sono conservati separatamente in
`analysis_outputs/ingredient_selection/phase3-d1-v3/inclusion_d6_v1/`;
`metrics.py` e gli artefatti D4 restano immutati. La politica è esplicitamente
post-outcome; [decisione](docs/project_objective/model_comparison_methodology.md#phase-3-d6--held-out-quality-inclusion-policy),
[risultato revisionato](docs/experiment_results/phase3_d1_v3_d6_profile.md) e
[contratto](docs/implementation_details/ingredient_selection.md#d6-saved-score-inclusion-review)
sono le fonti autorevoli. Il report di idoneità non esporta metadata selezionati.

La proiezione P6 è conservata in
[`src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json`](src/ingredient_selection/resources/ingredients_selected_v5_d6_v1.json)
e riprodotta da `scripts/ingredient_selection/export_projection.py`, tramite
`src/ingredient_selection/projection.py`. Contiene ordine selezionato, indici
nel vocabolario originale, gruppi esclusi/incerti, motivi e hash delle evidenze.
L'esportatore usa solo artefatti D6 approvati e libreria standard; non legge
metadata o predizioni e rifiuta di sovrascrivere contenuti diversi. È una
definizione esplicita consumata dal runtime P7 tramite
`ExpConfig(dm_ingredient_projection="ingredients_selected_v5_d6_v1")`.
Il default `None` resta completo (165 label); la proiezione usa le 59 classi
nell'ordine salvato e preserva tutti i record, anche con target proiettato vuoto.
`src/ingredient_selection/runtime.py` verifica identità, hash e corrispondenza
con le colonne originali; DataModule, configurazione e checkpoint rifiutano
vocabolari, encoder o dimensioni incompatibili. Non viene creato un nuovo
metadata. I checkpoint light conservano l'identità in un campo dedicato e
i vecchi checkpoint senza proiezione mantengono il comportamento serializzato.
Per analizzare output completi usare `project_output_columns` con l'ordine
originale salvato; il comparatore separa i cohort per hash della proiezione.
`scripts/ingredient_selection/verify_runtime.py` verifica senza scritture la
parità sui soli train/val reali, senza inferenza o accesso al test. Il
[contratto P7](docs/implementation_details/ingredient_selection.md#p7-runtime-projection)
documenta API, verifiche e ritiro logico degli script storici, conservati
byte-identici per la retention 2.1c. Usare un nome esperimento distinto per
il task selezionato, senza tentare di convertire una run completa in ripresa.

`src/lightning/lgn_models.py` incapsula un `BaseModel` in un `LightningModule`. La configurazione predefinita usa `BCEWithLogitsLoss` per la classificazione multi-label, con sigmoid in fase di calcolo metriche/inferenza. Le metriche di default includono accuracy, precision, recall e Hamming distance con media weighted; F1 non è abilitata di default e mancano average precision, calibrazione e selezione esplicita delle soglie. Questa configurazione è legacy e non coincide con il protocollo deciso per il nuovo benchmark.

`src/lightning/lgn_trainers.py` fornisce:

- `BaseTrainer`: checkpoint monitorato su `val_loss`, logging CSV/TensorBoard/W&B e salvataggio della configurazione nel checkpoint;
- `BaseFasterTrainer`: variante con early stopping;
- `OptunaTrainer`: checkpoint più leggero e pruning Optuna su `val_loss`.

`src/training/` è il punto di ingresso canonico della pipeline di training. I suoi moduli orchestrano la costruzione/ripresa dell'esperimento, la preparazione del DataModule e l'avvio di Lightning; gli script esterni devono riusare queste API anziché ricostruire il flusso. `src/training/ingredient_selection.py` contiene il percorso specializzato ma canonico di Phase 3: loss pesata, AdamW e scheduler congelati, audit a modello fisso e resource gate. La logica riusabile di dati, metriche, isolamento del pilot, artefatti e analisi è in `src/ingredient_selection/`; i comandi sottili sono in `scripts/ingredient_selection/`.

`src/training/one_shot_exp.py` è l'entry point per un singolo esperimento. Crea o riprende la directory `experiments/<gruppo>/<nome>/trial_N`, prepara il DataModule, registra encoder e numero di classi nella configurazione, quindi avvia Lightning.

`src/training/htuning_exp.py` gestisce l'ottimizzazione con Optuna: persiste lo studio nel journal configurato, salva configurazione fissa e generatore di iperparametri, crea `trial_N` e copia il trial migliore in `trial_best`.

Le run W&B sono prodotte offline. Per sincronizzarle, `scripts/sync_wandb_runs.py` richiama `wandb beta sync` specificando esplicitamente `WANDB_ENTITY` e `WANDB_PROJECT_NAME`, definiti in `settings/config.py`; questo evita upload senza entity (URL del tipo `wandb.ai//...`) e aggira il problema del sync classico che rigenera ripetutamente `wandb-summary.json`. A ogni tentativo lo script assegna al caricamento un ID remoto nuovo, formato dall'ID locale e dal suffisso casuale `-sync-<UUID breve>`, così una run eliminata in precedenza non causa un errore HTTP 409. La run configuration PyCharm `sync_wandb_runs` carica `.env`, che deve contenere `WANDB_API_KEY`.

Le configurazioni sono oggetti `ExpConfig`, `HTunerExpConfig` e `HGeneratorConfig` in `src/commons/exp_config.py`. Consentono ai launcher di passare override con prefissi (ad esempio modello, trainer e DataModule) e di ricostruire esperimenti dai checkpoint.

## Launcher degli esperimenti

`scripts/launch_exps/selected_ingredients/` raccoglie i launcher sul vocabolario
condiviso D6 a 59 label e ospiterà quelli dei nuovi modelli dopo la qualificazione
Phase 5. `train_resnet.py` e `train_dinov2.py` trasferiscono le configurazioni
complete storiche dei trial 77 e 61, ma inizializzano nuovi modelli preaddestrati
con nuove head: 40 epoche, batch logico 128, nessun nuovo tuning o test predittivo.
Supportano `--dry-run`, nomi distinti e `--resume` esplicito con controlli di
proiezione, contratto e sorgenti. Il modulo canonico è
`src/training/selected_vocab.py`; non modifica né usa l'entry point one-shot.
I default fisici sono 128 per ResNet e 32/accumulo 4 per DINO, senza una nuova
misura di capacità CUDA. Comandi, persistenza e limiti metodologici sono nel
[contratto dei launcher](docs/implementation_details/selected_vocabulary_training.md).

Per creare o lanciare una nuova campagna sperimentale si aggiunge uno script in `scripts/launch_exps/`. Lo script definisce nome/directory dell'esperimento e i relativi override di configurazione, quindi richiama l'API pertinente di `src/training` (`make_one_shot_exp` oppure `make_htuning_exp`). Non costituisce una seconda pipeline di training.

Esempi già presenti:

- `resnet/train_resnets.py` per confronti tra ResNet e backbone pretrained;
- script equivalenti in `densenet/`;
- `dinov2/htuning_dinov2.py` per l'hyperparameter tuning di DINOv2 ViT-B/14 in linear probing;
- `htuning_*.py` per Optuna;
- `test_best_for_f1.py` per validare trial selezionati con F1 per ingrediente.

Gli script hanno parametri e nomi esperimento hard-coded: vanno verificati/adattati prima dell'esecuzione. Le note correnti in `dev_notes.md` indicano che i test DINOv2, one-shot e hyperparameter tuning sono ancora in corso.

## Visualizzazione e analisi

Il contratto degli artefatti sperimentali e i limiti delle analisi offline sono verificati in [`docs/implementation_details/experiment_artifacts.md`](docs/implementation_details/experiment_artifacts.md). L'audit del 14 settembre 2026 copre i 200 trial target-v5 ResNet/DINOv2 e dimostra l'estrazione locale degli istogrammi W&B su un trial per famiglia; configurazioni, CSV, TensorBoard, journal e checkpoint richiedono gestione esplicita di alias, riprese, loss pesate e scala AMP. Il logging opzionale per ingrediente nel Lightning model e il comparatore locale JSON/HTML sono implementati sotto [`src/lightning/lgn_models.py`](src/lightning/lgn_models.py) e [`scripts/analise_exp/compare_experiments/`](scripts/analise_exp/compare_experiments/). Il contratto mantenuto è [`docs/implementation_details/experiment_comparison.md`](docs/implementation_details/experiment_comparison.md), mentre [`docs/plans/experiment_comparison.md`](docs/plans/experiment_comparison.md) conserva il piano completato.

I risultati sperimentali revisionati hanno una sezione dedicata in [`docs/experiment_results/`](docs/experiment_results/README.md), separata da contratti, metodologia e output grezzi. Il primo record è il confronto storico [`basic_v5` ResNet18–DINOv2](docs/experiment_results/basic_v5_resnet_dinov2.md): identifica ResNet18 trial 77 come artefatto validation migliore, ma circoscrive il claim al confronto tra fine-tuning completo ResNet e linear probing DINOv2 con backbone congelato e mantiene differito il benchmark finale.

- `src/dashboards/dash/app.py` avvia una web app Dash sulla porta 8050.
- Le immagini in `data/` sono servite singolarmente dalla route Flask `/assets/data/<percorso>`; non deve esistere il precedente symlink `dash/static/assets/data`, perche WhiteNoise indicizza ricorsivamente gli asset all'avvio e blocca la dashboard sui dataset grandi.
- La pagina `model_visualization.py` carica esperimenti/checkpoint, mostra immagini e predizioni, e produce Grad-CAM e feature factorization per l'interpretabilità.
- La ricostruzione del DataModule nella dashboard passa da `src/dashboards/runtime.py`: usa le trasformazioni del modello caricato, conserva l'encoder salvato e verifica dimensioni/proiezione prima di costruire i dataset. Evita che il preprocessing generico sostituisca quello del modello, inclusa la normalizzazione dei pesi preaddestrati.
- `docs/technical_details/<area>/<titolo_problema>/explaination.md` raccoglie note tecniche permanenti su problemi diagnostici, cause, soluzione e verifica. Questi documenti affiancano i deep dive architetturali: il primo è `docs/technical_details/dino/gradcam_frozen_vit_tokens/explaination.md`, relativo a Grad-CAM e feature factorization con DINOv2 congelato.
- `start_tensorboard.py` serve gli esperimenti con TensorBoard.
- `start_optuna.py` avvia Optuna Dashboard; nel codice corrente la porta effettiva è 8055, mentre la costante di navigazione della Dash app è 8051: possibile incoerenza da verificare.
- `saved_plots/` conserva risultati e grafici storici, in particolare esperimenti ResNet del novembre 2024.

## Dipendenze e ambiente

Lo stack è Python con PyTorch 2.8, torchvision 0.23, Lightning 2.6, Optuna, scikit-learn, Dash, W&B, TensorBoard e librerie di analisi/visualizzazione. Il progetto è orientato a CUDA; `set_torch_constants()` abilita benchmark cuDNN, precisione matmul `medium` e multiprocess start method `spawn`.

### Ambiente locale e WSL verificato

> **Nota operativa:** il workspace predefinito del progetto è la copia interna al filesystem Linux di WSL, in `/root/projects/ingredient_recognition`. Salvo indicazione esplicita diversa, lettura e modifica del codice, comandi Git, test, training, launcher, documentazione e configurazioni PyCharm devono riferirsi a questa copia. Le copie sulle unità Windows o raggiunte da WSL tramite `/mnt/<unità>/...` non sono il workspace operativo predefinito, perché l'accesso attraverso il filesystem montato introduce l'overhead I/O osservato e aumenta il rischio di divergenza tra le copie.

Le run configuration PyCharm condivisibili sono raccolte in `pycharm_run_config/` (non necessariamente tracciate da Git). Sono parte del flusso operativo del progetto e definiscono directory di lavoro e interprete per i comandi comuni.

| Configurazione | Script | SDK/interprete |
| --- | --- | --- |
| `one_shot_exp` | `src/training/one_shot_exp.py` | `wsl_image_pytorch` |
| `app` | `src/dashboards/dash/app.py` | `image_pytorch` |
| `start_optuna` | `src/dashboards/start_optuna.py` | `image_pytorch` |
| `start_tensorboard` | `src/dashboards/start_tensorboard.py` | `image_pytorch` |
| `sync_wandb_runs` | `scripts/sync_wandb_runs.py` | `wsl_image_pytorch` |

Per alcuni modelli e run GPU va usato WSL2 con la distribuzione `Ubuntu-22.04` (Ubuntu 22.04.2 LTS). Al 22 luglio 2026 è stato verificato il seguente ambiente:

- kernel WSL: `6.18.33.2-microsoft-standard-WSL2`;
- GPU esposta in WSL: NVIDIA GeForce RTX 4060, 8188 MiB, driver 596.21;
- interprete corretto: `/root/miniconda3/envs/wsl_image_pytorch/bin/python`;
- Python 3.10.18; PyTorch 2.8.0+cu129, torchvision 0.23.0+cu129, torchaudio 2.8.0+cu129, Lightning 2.6.1;
- CUDA runtime PyTorch 12.9 e `torch.cuda.is_available()` restituisce `True`.

L'interprete di sistema WSL (`/usr/bin/python3`, Python 3.10.12) non include PyTorch: per eseguire codice del progetto in WSL occorre usare l'ambiente Conda `wsl_image_pytorch`, non quello di sistema.

Il file `.env` non è stato ispezionato perché può contenere segreti. I grandi pacchetti CUDA `.deb` e alcuni asset locali risultano non tracciati nel worktree alla data della ricognizione e non fanno parte di questa documentazione funzionale.

## Punti da approfondire o verificare

- Eseguire i rimanenti gate del piano [`docs/plans/additional_model_implementation.md`](docs/plans/additional_model_implementation.md): DICA-Net-S richiede integrazione diagnostica capability-aware; tutti e tre richiedono qualificazione CUDA e consumer reali. Gli artefatti originali MaxViT/EfficientNet e i tre adapter con Lightning/persistenza opt-in sono verificati. Il selettore non ne sostituisce i gate.
- Mantenere il contratto Data e rilanciare lo smoke isolato quando cambiano i consumer runtime; il piano Data e i gate 2.1c/2.4 sono completi.
- P7 è conclusa in `docs/plans/recognizable_ingredient_selection.md`: usare la proiezione esplicita nei futuri esperimenti autorizzati, senza modificare D4/D6, il default completo o la popolazione. Un'eventuale pulizia fisica necessita una decisione separata; le revisioni manuali del precedente P5 restano facoltative. Gli smoke reali indipendenti Data 2.4 sono ora verificati.
- Verificare e, se necessario, uniformare alcuni import che dipendono dalla directory di avvio (`config`, `models`, `data_processing` vs `settings.config`, `src.*`).
- Verificare la gestione di ripresa dello studio Optuna, che condivide un journal globale configurato in `experiments/journal.log`.
- Correggere o documentare la differenza fra porta Optuna dichiarata (8051) e quella usata dallo script (8055).
- Definire quali launcher richiedono formalmente l'SDK WSL `wsl_image_pytorch` oltre alla sincronizzazione W&B, invece dell'ambiente locale `image_pytorch`.

## Regola di aggiornamento

Aggiornare questo file quando cambia uno dei seguenti elementi: obiettivo del modello, struttura/semantica dei dati, pipeline di preprocessing, architetture, funzione di loss/metriche, entry point di training, persistenza degli esperimenti, dashboard, dipendenze operative o decisioni tecniche confermate. Per una modifica minore, integrare soltanto la sezione pertinente e aggiornare la data; per una modifica maggiore, aggiornare anche il diagramma del flusso e l'elenco dei punti da verificare. Se la modifica cambia anche lo stato o la pianificazione del progetto, aggiornare contestualmente `docs/general_plan.md` e, quando pertinente, il piano esecutivo in `docs/plans/` secondo le regole delle sezioni precedenti.
