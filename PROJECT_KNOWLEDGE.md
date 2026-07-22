# Ingredient Recognition — conoscenza del progetto

> Documento vivente per l'assistente e per chi lavora al repository. Va aggiornato a ogni modifica architetturale o funzionale rilevante, e quando si confermano nuove informazioni sul progetto.

**Ultimo aggiornamento:** 22 luglio 2026  
**Stato della ricognizione:** architettura e flusso principale verificati nel codice; risultati numerici, completezza dei dataset e percorsi di esecuzione non sono stati eseguiti in questa ricognizione.

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

`settings/config.py` definisce i percorsi assoluti a partire dalla root del repository. I dati elaborati vivono in `data/input`; quelli sorgente in `data/raw_input`.

Il `DataModule` di default usa `data/input/yummly`. Ogni split deve contenere un `metadata.json` e le immagini indicate dal campo `image`. Quando non esiste uno split `predict`, viene riutilizzato lo split `test`.

Per ogni ricetta il codice si aspetta, almeno, ingredienti nel campo `ingredients_ok`, un'immagine nel campo `image` e, se si filtra, la cucina nel campo `cuisine`. Il filtro ammette: american, chinese, french, greek, indian, italian, japanese, mexican, spanish, thai e all.

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

Le immagini sono normalmente ridimensionate a 224×224. Per i modelli generici il DataModule applica resize, `TrivialAugmentWide` in training e normalizzazione con statistiche del dataset. I wrapper torchvision usano le trasformazioni/normalizzazioni dei pesi ImageNet. DINOv2 usa normalizzazione ImageNet e crop dedicati.

## Addestramento e valutazione

`src/lightning/lgn_models.py` incapsula un `BaseModel` in un `LightningModule`. La configurazione predefinita usa `BCEWithLogitsLoss` per la classificazione multi-label, con sigmoid in fase di calcolo metriche/inferenza. Le metriche di default includono accuracy, precision, recall e Hamming distance; F1 complessiva e per-etichetta sono configurabili.

`src/lightning/lgn_trainers.py` fornisce:

- `BaseTrainer`: checkpoint monitorato su `val_loss`, logging CSV/TensorBoard/W&B e salvataggio della configurazione nel checkpoint;
- `BaseFasterTrainer`: variante con early stopping;
- `OptunaTrainer`: checkpoint più leggero e pruning Optuna su `val_loss`.

`src/training/` è il punto di ingresso canonico della pipeline di training. I suoi moduli orchestrano la costruzione/ripresa dell'esperimento, la preparazione del DataModule e l'avvio di Lightning; gli script esterni devono riusare queste API anziché ricostruire il flusso.

`src/training/one_shot_exp.py` è l'entry point per un singolo esperimento. Crea o riprende la directory `experiments/<gruppo>/<nome>/trial_N`, prepara il DataModule, registra encoder e numero di classi nella configurazione, quindi avvia Lightning.

`src/training/htuning_exp.py` gestisce l'ottimizzazione con Optuna: persiste lo studio nel journal configurato, salva configurazione fissa e generatore di iperparametri, crea `trial_N` e copia il trial migliore in `trial_best`.

Le configurazioni sono oggetti `ExpConfig`, `HTunerExpConfig` e `HGeneratorConfig` in `src/commons/exp_config.py`. Consentono ai launcher di passare override con prefissi (ad esempio modello, trainer e DataModule) e di ricostruire esperimenti dai checkpoint.

## Launcher degli esperimenti

Per creare o lanciare una nuova campagna sperimentale si aggiunge uno script in `scripts/launch_exps/`. Lo script definisce nome/directory dell'esperimento e i relativi override di configurazione, quindi richiama l'API pertinente di `src/training` (`make_one_shot_exp` oppure `make_htuning_exp`). Non costituisce una seconda pipeline di training.

Esempi già presenti:

- `resnet/train_resnets.py` per confronti tra ResNet e backbone pretrained;
- script equivalenti in `densenet/`;
- `htuning_*.py` per Optuna;
- `test_best_for_f1.py` per validare trial selezionati con F1 per ingrediente.

Gli script hanno parametri e nomi esperimento hard-coded: vanno verificati/adattati prima dell'esecuzione. Le note correnti in `dev_notes.md` indicano che i test DINOv2, one-shot e hyperparameter tuning sono ancora in corso.

## Visualizzazione e analisi

- `src/dashboards/dash/app.py` avvia una web app Dash sulla porta 8050.
- La pagina `model_visualization.py` carica esperimenti/checkpoint, mostra immagini e predizioni, e produce Grad-CAM e feature factorization per l'interpretabilità.
- `start_tensorboard.py` serve gli esperimenti con TensorBoard.
- `start_optuna.py` avvia Optuna Dashboard; nel codice corrente la porta effettiva è 8055, mentre la costante di navigazione della Dash app è 8051: possibile incoerenza da verificare.
- `saved_plots/` conserva risultati e grafici storici, in particolare esperimenti ResNet del novembre 2024.

## Dipendenze e ambiente

Lo stack è Python con PyTorch 2.8, torchvision 0.23, Lightning 2.6, Optuna, scikit-learn, Dash, W&B, TensorBoard e librerie di analisi/visualizzazione. Il progetto è orientato a CUDA; `set_torch_constants()` abilita benchmark cuDNN, precisione matmul `medium` e multiprocess start method `spawn`.

Il file `.env` non è stato ispezionato perché può contenere segreti. I grandi pacchetti CUDA `.deb` e alcuni asset locali risultano non tracciati nel worktree alla data della ricognizione e non fanno parte di questa documentazione funzionale.

## Punti da approfondire o verificare

- Eseguire una prova end-to-end su un piccolo subset per confermare i comandi di lancio e le versioni correnti delle dipendenze.
- Stabilire quale dataset e quale campo etichetta siano attualmente canonici (Yummly/`ingredients_ok` oppure Recipes1M/`ingredients_ner`).
- Verificare e, se necessario, uniformare alcuni import che dipendono dalla directory di avvio (`config`, `models`, `data_processing` vs `settings.config`, `src.*`).
- Verificare la gestione di ripresa dello studio Optuna, che condivide un journal globale configurato in `experiments/journal.log`.
- Correggere o documentare la differenza fra porta Optuna dichiarata (8051) e quella usata dallo script (8055).

## Regola di aggiornamento

Aggiornare questo file quando cambia uno dei seguenti elementi: obiettivo del modello, struttura/semantica dei dati, pipeline di preprocessing, architetture, funzione di loss/metriche, entry point di training, persistenza degli esperimenti, dashboard, dipendenze operative o decisioni tecniche confermate. Per una modifica minore, integrare soltanto la sezione pertinente e aggiornare la data; per una modifica maggiore, aggiornare anche il diagramma del flusso e l'elenco dei punti da verificare.
