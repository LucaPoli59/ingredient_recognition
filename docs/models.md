# Modelli di visione

Questa pagina descrive l'implementazione dei modelli disponibili in `src/models` e il loro contratto con la pipeline di training. Il problema resta una classificazione multi-label: ciascun modello emette un vettore di `num_classes` **logit**, senza sigmoid finale. La conversione in probabilità e l'applicazione di `BCEWithLogitsLoss` sono responsabilità del modulo Lightning.

La documentazione privilegia gli aspetti di integrazione e le decisioni architetturali del repository; non ripete la teoria introduttiva di CNN, residual connection o transformer. Le sezioni contrassegnate come *da ampliare* sono intenzionalmente solo una traccia iniziale.

## Approfondimenti architetturali

Per i dettagli di machine learning, della struttura interna delle reti e della ricerca di riferimento, consultare i documenti nella cartella [`docs/models_deepdive/`](models_deepdive/). Al momento è disponibile il deep dive su [DINOv2 ViT-B/14](models_deepdive/dinov2.md); i nuovi approfondimenti sui restanti modelli saranno aggiunti nella stessa cartella.

## Contratto comune: `BaseModel`

`BaseModel` è l'interfaccia comune di tutti i modelli di visione. Conserva `num_classes`, la dimensione di input quadrata e i builder delle trasformazioni; espone inoltre `transform_aug` e `transform_plain`, usati dal DataModule rispettivamente per train e validazione/inferenza. Le trasformazioni sono quindi parte della configurazione serializzabile del modello e non un dettaglio esterno della run.

Ogni sottoclasse deve esporre `conv_target_layer` e `classifier_target_layer`. Questi hook sono consumati dalla dashboard di visualizzazione (per esempio Grad-CAM) e devono riferirsi a moduli effettivamente attraversati dal `forward`.

### Serializzazione e ricostruzione

`to_config()` registra il tipo concreto e i parametri comuni; `load_from_config()` valida che il tipo richiesto coincida con la classe che sta ricostruendo l'oggetto. I wrapper pretrained estendono questo payload con le proprie opzioni. Le callable di trasformazione non standard restano oggetti Python nella configurazione: la loro persistenza richiede quindi il normale meccanismo di checkpoint/configurazione del progetto, non una serializzazione JSON portabile autonoma.

### Layer-wise pretraining nei ResNet custom

Il protocollo opzionale di layer-wise pretraining (LP) è implementato in `BaseModel` e concretizzato solo dalla famiglia ResNet custom. Con `lp_phase >= 0`, i blocchi `layer1`–`layer4` vengono inizialmente sostituiti da `Identity`; resta allenabile il trunk disponibile e una testata compatibile con il suo numero di canali. Ogni chiamata a `lp_phase_step()` congela l'ultimo stadio allenato, installa lo stadio successivo e ricrea il classificatore. Dopo l'ultima fase, tutti i parametri vengono nuovamente scongelati e `lp_phase` passa a `-1`.

Questo meccanismo modifica la topologia effettiva durante l'addestramento: non è un semplice scheduler del learning rate. Checkpoint e ripresa devono perciò mantenere coerente `lp_phase`; le famiglie DenseNet e i wrapper torchvision non lo supportano.

## ResNet custom

Le classi `ResnetLikeV1`, `ResnetLikeV1LVariant` e `ResnetLikeV2` condividono lo stem `7×7, stride 2` seguito da batch normalization, ReLU e max-pooling. La testata è sempre `AdaptiveAvgPool2d(1) → Flatten → Linear`, perciò il numero di classi può cambiare senza dipendere dalla risoluzione spaziale finale.

### Blocchi e progressione dei canali

`ResnetLikeV1` replica la profondità di ResNet-18: due `BasicBlock` per ciascuno dei quattro stadi. Un `BasicBlock` usa due convoluzioni `3×3`; quando stride o canali non coincidono, il ramo identity diventa una proiezione `1×1` con batch normalization. I canali seguono `64 → 64 → 128 → 256 → 512`, con downsampling all'ingresso degli ultimi tre stadi.

`ResnetLikeV1LVariant` mantiene la stessa configurazione ma sostituisce l'attivazione dei blocchi con `LeakyReLU`. Lo stem resta invariato, quindi la variante agisce soltanto sulla non linearità dei rami residui.

`ResnetLikeV2` segue la struttura ResNet-50 (`3, 4, 6, 3` blocchi), con `BottleneckBlock` a espansione quattro: `1×1` di compressione, `3×3` per l'estrazione, `1×1` di espansione. Nei costruttori la testata viene inizialmente creata con 512 feature, ma `_make_classifier()` applica `LAYER_EXPANSION = 4`; riceve quindi le 2048 feature prodotte dall'ultimo stadio.

### Implicazioni operative

I modelli custom usano i builder di trasformazioni generici del progetto, non quelli legati ai pesi ImageNet. Il loro `conv_target_layer` è il blocco finale dell'ultimo stadio (o il blocco sottostante, anche in LP), mentre il target del classificatore è l'ultimo `Linear`. Sono pertanto direttamente utilizzabili dagli strumenti di interpretabilità della dashboard.

## DenseNet custom

`DensenetLikeV1` e `DensenetLikeV2` condividono lo stem dei ResNet custom, ma sostituiscono la composizione residuale con concatenazioni di feature. Ogni `DenseLayer` applica la sequenza pre-attivata `BN → ReLU → 1×1 → BN → ReLU → 3×3` e concatena alla propria uscita l'input originale. Il growth rate è 32: ogni layer aggiunge esattamente 32 canali al tensore dello stesso stadio.

### Compressione e dimensioni

La convoluzione `1×1` interna lavora a 128 canali (`growth_rate × 4`), limitando il costo della successiva `3×3`. Dopo ciascuno dei primi tre dense block, `TransitionLayer` esegue `BN → ReLU → 1×1 → AvgPool2d(2)` e dimezza i canali (`reduction_factor = 0.5`) oltre alla risoluzione. L'ultima normalizzazione e ReLU precedono global average pooling e classificatore lineare.

| Modello | Layer per dense block | Canali dopo i dense block | Canali dopo le transition |
| --- | --- | --- | --- |
| `DensenetLikeV1` | 6, 12, 24, 16 | 256, 512, 1024, 1024 | 128, 256, 512 |
| `DensenetLikeV2` | 6, 12, 48, 32 | 256, 512, 1792, 1920 | 128, 256, 896 |

L'uso di concatenazione preserva feature di tutti i layer precedenti, ma aumenta la pressione di memoria, soprattutto nel terzo e quarto blocco della V2. Non è previsto LP per questi modelli; il parametro ricevuto viene neutralizzato nel costruttore base.

## DINOv2 ViT-B/14 con registri

`DinoV2B14` carica tramite `torch.hub` il modello `dinov2_vitb14_reg_lc` dal repository `facebookresearch/dinov2`. Il backbone è un Vision Transformer base con patch `14×14` e token di registro; il suffisso `_lc` seleziona la variante dotata di linear classifier. La classe sostituisce `model.linear_head` con un nuovo `Linear` avente l'output pari a `num_classes`, così la testata del checkpoint upstream non viene riusata per gli ingredienti.

### Congelamento e fine-tuning

Per default `freeze_backbone=True`: `freeze_backbone()` imposta `requires_grad=False` sui parametri del backbone, lasciando allenabile la nuova linear head. `unfreeze_backbone()` abilita il fine-tuning completo in un secondo momento. Il limite dichiarato di batch è 32 attraverso `max_allowed_batch_size`, utile alla pipeline per evitare configurazioni troppo grandi per la GPU.

Il parametro `pretrained` viene conservato nella configurazione, ma l'implementazione corrente chiama comunque `torch.hub.load(...)` senza usarlo per scegliere pesi o architettura: il caricamento del backbone è quindi sempre quello definito da torch.hub. Questo è un dettaglio importante se si vuole un vero avvio da pesi casuali.

### Preprocessing e interpretabilità

DINOv2 usa builder dedicati (`transform_*_dino`). Se viene passata una funzione di augmentation, il train abilita `random_crop=True`, mentre validation/inferenza usa `random_crop=False`; senza override vengono impiegati direttamente i builder configurati. Per gli strumenti di visualizzazione, il target assimilato a uno strato convoluzionale è `backbone.norm`, mentre il classificatore è `linear_head`.

## Wrapper torchvision ResNet

*Da ampliare.* `Resnet18` e `Resnet50` costruiscono le rispettive architetture torchvision, opzionalmente con i pesi `DEFAULT`, e sostituiscono `model.fc` con una proiezione verso `num_classes`. Impostano inoltre trasformazioni compatibili con i pesi ImageNet e pubblicano `layer4[-1]` come target visuale.

## Wrapper torchvision DenseNet

*Da ampliare.* `Densenet121` e `Densenet201` sostituiscono il classificatore torchvision dopo l'estrattore di feature DenseNet. I target di interpretabilità sono l'ultimo modulo di `model.features` e il classificatore lineare. Questa sezione sarà estesa con le implicazioni delle varianti di costruttore e delle trasformazioni pretrained.

## Modelli dummy

*Da ampliare.* `DummyModel` e `DummyBNModel` sono reti di test con tre blocchi convoluzionali e max-pooling; la seconda inserisce batch normalization. Sono utili per validare training, shape e dashboard senza il costo dei backbone principali, non come baseline architetturale competitiva.

## Scheduler presenti in `src/models`

*Da ampliare.* `WarmStartReduceOnPlateau` e `ConstantStartReduceOnPlateau` derivano da `ReduceLROnPlateau` e aggirano l'incompatibilità storica fra `SequentialLR` e Lightning. Il primo interpola il learning rate da `warm_start` a `warm_stop` (lineare o con `tanh`) prima di delegare alla logica plateau; il secondo mantiene l'LR iniziale nella fase di attesa.

## Riferimenti al codice

- `src/models/commons.py`: contratto comune, trasformazioni e LP.
- `src/models/resnet.py`: ResNet custom e wrapper torchvision.
- `src/models/densenet.py`: DenseNet custom e wrapper torchvision.
- `src/models/dinov2.py`: wrapper DINOv2 e gestione del backbone congelato.
- `src/models/dummy.py`: modelli minimali per test.
- `src/models/custom_schedulers.py`: scheduler custom.
