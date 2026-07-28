# Dettagli tecnici

**Data di creazione:** 29 luglio 2026

Questa cartella raccoglie note tecniche approfondite su concetti, limiti e problemi emersi nel progetto. Lo scopo è preservare il ragionamento di machine learning, reti neurali, elaborazione dei dati o sistemi che sta dietro a una decisione tecnica, così da renderlo riutilizzabile e verificabile in futuro.

Questi documenti non sono guide operative per applicare una patch né changelog delle modifiche al codice. Possono citare un caso del repository come motivazione, ma devono spiegare prima il fenomeno in termini generali: assunzioni, rappresentazioni, formule o shape rilevanti, conseguenze e limiti interpretativi.

## Struttura

Ogni nota segue il percorso:

```text
docs/technical_details/<area>/<titolo_problema>/explaination.md
```

`<area>` identifica il dominio tecnico, per esempio `dino`, `data`, `lightning` o `dashboard`. `<titolo_problema>` deve essere breve, descrittivo e usare `snake_case`.

Ogni nuovo documento deve riportare subito sotto il titolo la data in cui è stato creato, nel formato:

```markdown
**Data di creazione:** GG mese AAAA
```

## Contenuto atteso

Una nota dovrebbe includere, quando pertinente:

- contesto e domanda tecnica;
- modello concettuale, matematico o architetturale;
- causa del fenomeno e condizioni necessarie perché si manifesti;
- implicazioni per il progetto e limiti della spiegazione;
- un sunto finale.

Il codice interessato può essere citato come riferimento, ma le istruzioni di implementazione dettagliate appartengono ai commenti, alle pull request o alla documentazione funzionale dei moduli.

## Documenti disponibili

- [`dino/gradcam_frozen_vit_tokens/explaination.md`](dino/gradcam_frozen_vit_tokens/explaination.md): interpretabilità differenziabile, Grad-CAM e rappresentazioni token-based in DINOv2.
