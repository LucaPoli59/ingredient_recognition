# Deep dive sui modelli

**Data di creazione:** 29 luglio 2026

Questa cartella contiene approfondimenti tecnici sulle architetture di machine learning disponibili nel progetto. Ogni documento descrive il modello in sé: rappresentazioni interne, layer e flusso del forward, obiettivi di pretraining quando rilevanti, scelte architetturali, funzione della head downstream e implicazioni per il task di riconoscimento degli ingredienti.

L'obiettivo è andare oltre la panoramica in [`../models.md`](../models.md) senza ripetere le nozioni introduttive di reti neurali. I deep dive devono basarsi sulle implementazioni effettivamente usate dal repository e, quando trattano architetture pretrained o metodi di ricerca, fare riferimento alle fonti primarie.

## Struttura

Ogni approfondimento è un file Markdown nominato come il modello:

```text
docs/models_deepdive/<nome_modello>.md
```

Il nome deve essere in minuscolo e descrittivo, ad esempio `dinov2.md`, `resnet.md` o `densenet.md`.

Ogni nuovo approfondimento deve riportare subito sotto il titolo la data in cui è stato creato, nel formato:

```markdown
**Data di creazione:** GG mese AAAA
```

## Contenuto atteso

Un deep dive dovrebbe includere, quando pertinente:

- variante precisa del modello e ruolo nella pipeline;
- flusso dei tensori, shape e rappresentazioni interne;
- struttura dei blocchi e dei layer principali;
- obiettivo di pretraining o addestramento originario;
- adattamento al classificatore multi-label di ingredienti;
- trade-off, limiti e conseguenze per training, inferenza e interpretabilità;
- fonti primarie e riferimenti al codice locale.

I problemi tecnici trasversali — ad esempio un errore di gradienti, una limitazione di una libreria o una scelta di integrazione — non appartengono a questa cartella: vanno documentati in [`../technical_details/`](../technical_details/).

## Documenti disponibili

- [`dinov2.md`](dinov2.md): ViT-B/14 con register token, pretraining DINO+iBOT, head `_lc` e integrazione nella pipeline multi-label.
