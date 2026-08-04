# Ingredient Recognition — conoscenza del progetto

> Documento vivente per l'assistente e per chi lavora al repository. Va aggiornato a ogni modifica architetturale o funzionale rilevante, e quando si confermano nuove informazioni sul progetto.

**Ultimo aggiornamento:** 3 agosto 2026
**Stato della ricognizione:** architettura e flusso principale verificati nel codice; il dataset Yummly è stato analizzato integralmente su metadata e 65.146 immagini, con audit riproducibili di target, duplicati, qualità visiva e contaminazione degli split.

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

Il `DataModule` di default usa `data/input/yummly`. I metadata restano in `train/`, `val/` e `test/`, mentre tutte le immagini sono risolte da `imgs/standard/`; quando non esiste uno split `predict`, viene riutilizzato `test`. La generazione nuova di default è `ingredients_target_v1_metadata.json` con `feature_label="ingredients_target"`; `metadata.json` e `sel_ing_2410_metadata.json` con `ingredients_ok` restano generazioni legacy immutabili.

Per ogni ricetta il codice si aspetta almeno il campo selezionato da `feature_label`, un'immagine nel campo `image` e, se si filtra, la cucina nel campo `cuisine`. Il default corrente è `ingredients_ok`; per le nuove configurazioni diventerà `ingredients_target`, derivato da `ingredients`, mentre gli esperimenti storici manterranno esplicitamente `ingredients_ok`. Il filtro ammette: american, chinese, french, greek, indian, italian, japanese, mexican, spanish, thai e all.

### Preparazione

- `src/raw2input/yummly/recipes_merge.py` riunisce le ricette raw di Yummly in `all_recipes.json`.
- `src/raw2input/yummly/sort_recipes_as_img.py` ordina le ricette in base alle immagini disponibili.
- `src/raw2input/yummly/creation.py` crea gli split (seed 42; val/test 8% ciascuno), copia le immagini disponibili e genera i relativi metadata.
- `src/raw2input/recipes1M/extraction.py` è uno script parzialmente operativo per campionare ricette, scaricarne immagini e preparare Recipes1M; contiene ancora porzioni commentate.
- `src/raw2input/compute_img_stats.py` calcola media e deviazione standard RGB dello split di training e salva `train_images_stats.csv`. Il file è richiesto dal DataModule base per le trasformazioni standard.

### Etichette e split

`ImagesRecipesBaseDataModule` carica tutti gli split, applica il filtro per cucina e adatta/usa un encoder multi-label. L'encoder predefinito è `MultiLabelBinarizerRobust`; produce target multi-hot e gestisce classi non osservate in uno split. I pesi di classe sono calcolati dalle frequenze dello split train e possono essere usati dalla loss.

Sono presenti dataset/encoder ulteriori per one-vs-all, classificazione multi-classe, sequenze di ingredienti con token speciali, masking e flavour: sono secondari rispetto alla pipeline immagini → ingredienti.

## Modelli e preprocessing

Tutti i modelli di visione discendono da `BaseModel`, che centralizza configurazione, serializzazione e definizione delle trasformazioni train/validation.

- `src/models/resnet.py`: ResNet custom simili a ResNet-18/50 e wrapper torchvision per ResNet18 e ResNet50, con teste adattate al numero di ingredienti.
- `src/models/densenet.py`: DenseNet custom e wrapper torchvision DenseNet121/DenseNet201.
- `src/models/dinov2.py`: DINOv2 ViT-B/14 con head lineare sostituita; usa `torch.hub` per caricare `facebookresearch/dinov2` e può congelare il backbone (default).
- `src/models/dummy.py`: modelli minimi per test.

La struttura, la lingua e la metodologia di scrittura della documentazione sono definite in `docs/README.md`. Tutti i documenti sotto `docs/` devono essere scritti in inglese. Per approfondimenti tecnici sulle architetture e sulla ricerca di riferimento, consultare `docs/models_deepdive/`. Al momento è disponibile `docs/models_deepdive/dinov2.md`, dedicato a DINOv2 ViT-B/14; la panoramica dei modelli è in `docs/implementation_details/models.md`.

L'obiettivo di ricerca e il relativo audit dei dati sono formalizzati in `docs/project_objective/`. Il dataset attivo è esclusivamente Yummly: l'audit ha dimostrato che le etichette `ingredients_ok` correnti non sono esattamente riproducibili e ha quantificato collisioni lessicali, duplicati tra split, rumore visivo e limiti di osservabilità. Le decisioni vincolanti sono in `docs/project_objective/benchmark_decisions.md`: mantenere `feature_label` configurabile con nuovo default `ingredients_target`, rigenerare quel campo da `ingredients` con regole deterministiche, usare solo controlli automatici sulle immagini, raggruppare soltanto immagini byte-identiche tramite SHA-256, creare uno split 80/10/10 bilanciato, evitare manifest duplicati, preservare senza riscriverli soltanto gli esperimenti legacy che saranno selezionati, rimuovere `<UNK>` dai nuovi output multi-label mantenendolo negli artefatti legacy selezionati, usare mAP macro e micro F1 come metriche primarie abbinate e calibrare/sogliare soltanto sulla validation.

Lo stato di avanzamento dell'intero progetto di tesi è mantenuto in `docs/general_plan.md`. Il tracker conserva lo storico ed è organizzato nelle macro-sezioni fondazione, dati, selezione degli ingredienti, ricerca e implementazione di modelli aggiuntivi, training e hyperparameter tuning, confronto dei risultati e scrittura della tesi. I piani esecutivi delle implementazioni concrete sono mantenuti separatamente in `docs/plans/`. La priorità corrente è l'audit del vocabolario candidato (2.2a); la compatibilità legacy è deferred senza modificare gli artefatti salvati.

Il piano esecutivo attivo della fase Data è `docs/plans/yummly_data_phase.md`. Copre i Work package 2.1b–2.4: store immagini condiviso, compatibilità degli esperimenti storici, standardizzatore `ingredients_target`, audit del vocabolario (2.2a), rafforzamento approvato dell'estrattore (2.2b), split deterministico con gruppi SHA-256 esatti e integrazione runtime senza `<UNK>` nei nuovi output multi-label. La compatibilità legacy (2.1c) è deferred finché non saranno scelti gli esperimenti storici da mantenere. La generazione iniziale con 209 target è una candidata validata, non un benchmark congelato: 2.2a ne analizzerà supporto, co-occorrenze, varianti lessicali e collisioni e presenterà i risultati per discuterli; nessuna regola o metadata sarà modificata prima di tale decisione. Prima di modificare loader, layout Yummly, metadata o builder occorre leggere sia il piano generale sia questo piano esecutivo e mantenerne sincronizzati i tracker.

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

Le immagini sono normalmente ridimensionate a 224×224. Per i modelli generici il DataModule applica resize, `TrivialAugmentWide` in training e normalizzazione con statistiche del dataset. I wrapper torchvision usano le trasformazioni/normalizzazioni dei pesi ImageNet. DINOv2 usa normalizzazione ImageNet e crop dedicati.

## Addestramento e valutazione

`src/lightning/lgn_models.py` incapsula un `BaseModel` in un `LightningModule`. La configurazione predefinita usa `BCEWithLogitsLoss` per la classificazione multi-label, con sigmoid in fase di calcolo metriche/inferenza. Le metriche di default includono accuracy, precision, recall e Hamming distance con media weighted; F1 non è abilitata di default e mancano average precision, calibrazione e selezione esplicita delle soglie. Questa configurazione è legacy e non coincide con il protocollo deciso per il nuovo benchmark.

`src/lightning/lgn_trainers.py` fornisce:

- `BaseTrainer`: checkpoint monitorato su `val_loss`, logging CSV/TensorBoard/W&B e salvataggio della configurazione nel checkpoint;
- `BaseFasterTrainer`: variante con early stopping;
- `OptunaTrainer`: checkpoint più leggero e pruning Optuna su `val_loss`.

`src/training/` è il punto di ingresso canonico della pipeline di training. I suoi moduli orchestrano la costruzione/ripresa dell'esperimento, la preparazione del DataModule e l'avvio di Lightning; gli script esterni devono riusare queste API anziché ricostruire il flusso.

`src/training/one_shot_exp.py` è l'entry point per un singolo esperimento. Crea o riprende la directory `experiments/<gruppo>/<nome>/trial_N`, prepara il DataModule, registra encoder e numero di classi nella configurazione, quindi avvia Lightning.

`src/training/htuning_exp.py` gestisce l'ottimizzazione con Optuna: persiste lo studio nel journal configurato, salva configurazione fissa e generatore di iperparametri, crea `trial_N` e copia il trial migliore in `trial_best`.

Le run W&B sono prodotte offline. Per sincronizzarle, `scripts/sync_wandb_runs.py` richiama `wandb beta sync` specificando esplicitamente `WANDB_ENTITY` e `WANDB_PROJECT_NAME`, definiti in `settings/config.py`; questo evita upload senza entity (URL del tipo `wandb.ai//...`) e aggira il problema del sync classico che rigenera ripetutamente `wandb-summary.json`. A ogni tentativo lo script assegna al caricamento un ID remoto nuovo, formato dall'ID locale e dal suffisso casuale `-sync-<UUID breve>`, così una run eliminata in precedenza non causa un errore HTTP 409. La run configuration PyCharm `sync_wandb_runs` carica `.env`, che deve contenere `WANDB_API_KEY`.

Le configurazioni sono oggetti `ExpConfig`, `HTunerExpConfig` e `HGeneratorConfig` in `src/commons/exp_config.py`. Consentono ai launcher di passare override con prefissi (ad esempio modello, trainer e DataModule) e di ricostruire esperimenti dai checkpoint.

## Launcher degli esperimenti

Per creare o lanciare una nuova campagna sperimentale si aggiunge uno script in `scripts/launch_exps/`. Lo script definisce nome/directory dell'esperimento e i relativi override di configurazione, quindi richiama l'API pertinente di `src/training` (`make_one_shot_exp` oppure `make_htuning_exp`). Non costituisce una seconda pipeline di training.

Esempi già presenti:

- `resnet/train_resnets.py` per confronti tra ResNet e backbone pretrained;
- script equivalenti in `densenet/`;
- `dinov2/htuning_dinov2.py` per l'hyperparameter tuning di DINOv2 ViT-B/14 in linear probing;
- `htuning_*.py` per Optuna;
- `test_best_for_f1.py` per validare trial selezionati con F1 per ingrediente.

Gli script hanno parametri e nomi esperimento hard-coded: vanno verificati/adattati prima dell'esecuzione. Le note correnti in `dev_notes.md` indicano che i test DINOv2, one-shot e hyperparameter tuning sono ancora in corso.

## Visualizzazione e analisi

- `src/dashboards/dash/app.py` avvia una web app Dash sulla porta 8050.
- Le immagini in `data/` sono servite singolarmente dalla route Flask `/assets/data/<percorso>`; non deve esistere il precedente symlink `dash/static/assets/data`, perche WhiteNoise indicizza ricorsivamente gli asset all'avvio e blocca la dashboard sui dataset grandi.
- La pagina `model_visualization.py` carica esperimenti/checkpoint, mostra immagini e predizioni, e produce Grad-CAM e feature factorization per l'interpretabilità.
- `docs/technical_details/<area>/<titolo_problema>/explaination.md` raccoglie note tecniche permanenti su problemi diagnostici, cause, soluzione e verifica. Questi documenti affiancano i deep dive architetturali: il primo è `docs/technical_details/dino/gradcam_frozen_vit_tokens/explaination.md`, relativo a Grad-CAM e feature factorization con DINOv2 congelato.
- `start_tensorboard.py` serve gli esperimenti con TensorBoard.
- `start_optuna.py` avvia Optuna Dashboard; nel codice corrente la porta effettiva è 8055, mentre la costante di navigazione della Dash app è 8051: possibile incoerenza da verificare.
- `saved_plots/` conserva risultati e grafici storici, in particolare esperimenti ResNet del novembre 2024.

## Dipendenze e ambiente

Lo stack è Python con PyTorch 2.8, torchvision 0.23, Lightning 2.6, Optuna, scikit-learn, Dash, W&B, TensorBoard e librerie di analisi/visualizzazione. Il progetto è orientato a CUDA; `set_torch_constants()` abilita benchmark cuDNN, precisione matmul `medium` e multiprocess start method `spawn`.

### Ambiente locale e WSL verificato

Le run configuration PyCharm condivisibili sono raccolte in `pycharm_run_config/` (non necessariamente tracciate da Git). Sono parte del flusso operativo del progetto e definiscono directory di lavoro e interprete per i comandi comuni.

| Configurazione | Script | SDK/interprete |
| --- | --- | --- |
| `one_shot_exp` | `src/training/one_shot_exp.py` | SDK di progetto `image_pytorch` |
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

- Eseguire una prova end-to-end su un piccolo subset per confermare i comandi di lancio e le versioni correnti delle dipendenze.
- Implementare la fase Data definita in `docs/plans/yummly_data_phase.md`: store immagini condiviso, adattatore legacy in memoria, standardizzazione `ingredients_target`, split deterministico con soli gruppi SHA-256 esatti e integrazione runtime.
- Verificare e, se necessario, uniformare alcuni import che dipendono dalla directory di avvio (`config`, `models`, `data_processing` vs `settings.config`, `src.*`).
- Verificare la gestione di ripresa dello studio Optuna, che condivide un journal globale configurato in `experiments/journal.log`.
- Correggere o documentare la differenza fra porta Optuna dichiarata (8051) e quella usata dallo script (8055).
- Definire quali launcher richiedono formalmente l'SDK WSL `wsl_image_pytorch` oltre alla sincronizzazione W&B, invece dell'ambiente locale `image_pytorch`.

## Regola di aggiornamento

Aggiornare questo file quando cambia uno dei seguenti elementi: obiettivo del modello, struttura/semantica dei dati, pipeline di preprocessing, architetture, funzione di loss/metriche, entry point di training, persistenza degli esperimenti, dashboard, dipendenze operative o decisioni tecniche confermate. Per una modifica minore, integrare soltanto la sezione pertinente e aggiornare la data; per una modifica maggiore, aggiornare anche il diagramma del flusso e l'elenco dei punti da verificare. Se la modifica cambia anche lo stato o la pianificazione del progetto, aggiornare contestualmente `docs/general_plan.md` e, quando pertinente, il piano esecutivo in `docs/plans/` secondo le regole delle sezioni precedenti.
