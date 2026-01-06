#QUESTO VA ESEGUITO USANDO L'ENVIRONMENT "TEXT_TORCH" CHE HA PYTORCH, TORCHTEXT (che non è compatibile con la versioen di torchvision) E SPACY

import torch
import os
import json
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
import torchtext
import lightning as lgn

from src_scratches.recipes_standardization.seq2seq_model import Encoder, Decoder, Seq2Seq, Seq2SeqDataset, Seq2SeqLightning, Seq2SeqTrainer
from settings.config import RECIPES_PATH, METADATA_FILENAME
from src_scratches.recipes_standardization.dictionaries import units, quantities_dict, modifiers


def synthetic_gen(num_samples):
    x = [0] * num_samples
    y = [0] * num_samples

    for i in range(num_samples):

        rnd_qty_str, rnd_qty_int = random.choice(list(quantities_dict.items()))

        no_units_present = random.choice([False, False, False, False, True])
        rnd_unit = random.choice(units)

        rnd_mod_present = random.choice([None, None, True])
        rnd_mod = random.choice(modifiers)

        rnd_ing = random.choice(raw_ingredients)

        # Build the output string, Y
        # e.g. {"qty": 36, "unit": "count", "item": "eggs", "mod": "scrambled"}
        if no_units_present:
            rnd_unit = 'count'  # For purposes of building Y

        if rnd_mod_present:
            y[i] = f'{{ qty: {rnd_qty_int} , unit: {rnd_unit} , item: {rnd_ing} , mod: {rnd_mod} }}'
        else:
            y[i] = f'{{ qty: {rnd_qty_int} , unit: {rnd_unit} , item: {rnd_ing} , mod: {None} }}'

        # Build the input string, X
        # e.g. "3 dozen scrambled eggs"
        mod_at_end = [False, True]
        rnd_mod_at_end = random.choice(mod_at_end)

        # avoiding double space
        if rnd_mod_present:
            if no_units_present:
                if rnd_mod_at_end:
                    x[i] = f'{rnd_qty_str} {rnd_ing} , {rnd_mod}'  # e.g. 3 eggs, scrambled
                else:
                    x[i] = f'{rnd_qty_str} {rnd_mod} {rnd_ing}'  # e.g. 3 scrambled eggs
            else:
                if rnd_mod_at_end:
                    x[i] = f'{rnd_qty_str} {rnd_unit} {rnd_ing} , {rnd_mod}'  # e.g. 3 cups eggs, scrambled
                else:
                    x[i] = f'{rnd_qty_str} {rnd_unit} {rnd_mod} {rnd_ing}'  # e.g. 3 cups scrambled eggs
        else:
            if no_units_present:
                x[i] = f'{rnd_qty_str} {rnd_ing}'  # e.g. 3 eggs
            else:
                x[i] = f'{rnd_qty_str} {rnd_unit} {rnd_ing}'  # e.g. 3 cups eggs

    return x, y

if __name__ == "__main__":

    recipes = json.load(open(os.path.join(RECIPES_PATH, METADATA_FILENAME), 'r'))
    raw_ingredients = pd.DataFrame(recipes)['ingredients'].explode().unique()

    N_SAMPLES = 100000
    TRAIN_SIZE = 0.95 # 95% of the data
    BATCH_SIZE = 256
    EPOCHS = 20
    NUM_WORKERS = os.cpu_count()
    data = synthetic_gen(N_SAMPLES)
    x, y = data

    en_nlp = spacy.load("en_core_web_sm")
    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"

    def tokenize_en(text, tokenizer=en_nlp, sos_token=sos_token, eos_token=eos_token):
        tokenized =  [token.text.lower() for token in tokenizer(text)]
        return [sos_token] + tokenized + [eos_token]

    def tokenize_whitespaces(text, sos_token=sos_token, eos_token=eos_token):
        tokenized = text.lower().split()
        return [sos_token] + tokenized + [eos_token]
    tokenize_en("3 cups scrambled eggs")

    min_freq = 1
    special_tokens = [sos_token, eos_token, unk_token, pad_token]

    x_tokens = [tokenize_whitespaces(i) for i in tqdm(x)]
    y_tokens = [tokenize_whitespaces(i) for i in tqdm(y)]

    train_len = int(len(x_tokens) * TRAIN_SIZE)

    x_train, y_train = x_tokens[:train_len], y_tokens[:train_len]
    x_val, y_val = x_tokens[train_len:], y_tokens[train_len:]

    x_vocab = torchtext.vocab.build_vocab_from_iterator(x_train, min_freq=min_freq, specials=special_tokens)
    x_vocab.set_default_index(x_vocab[unk_token])

    y_vocab = torchtext.vocab.build_vocab_from_iterator(y_train, min_freq=min_freq, specials=special_tokens)
    y_vocab.set_default_index(y_vocab[unk_token])

    assert x_vocab[pad_token] == y_vocab[pad_token]
    train_dataset = Seq2SeqDataset(x_train, y_train, x_vocab, y_vocab, pad_token=pad_token)
    val_dataset = Seq2SeqDataset(x_val, y_val, x_vocab, y_vocab, pad_token=pad_token)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                    pin_memory=True, persistent_workers=True, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                pin_memory=True, persistent_workers=True, collate_fn=val_dataset.collate_fn)


    input_dim = len(x_vocab)
    output_dim = len(y_vocab)
    encoder_embedding_dim = 128
    encoder_bidirectional = True
    decoder_embedding_dim = 128
    hidden_dim = 256
    n_layers = 2
    encoder_dropout = 0.0
    decoder_dropout = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        input_dim,
        encoder_embedding_dim,
        hidden_dim,
        n_layers,
        encoder_dropout,
        bidirectional=encoder_bidirectional,
    )

    decoder = Decoder(
        output_dim,
        decoder_embedding_dim,
        hidden_dim,
        n_layers * (2 if encoder_bidirectional else 1),
        decoder_dropout,
    )

    model = Seq2Seq(encoder, decoder, device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"The model has {count_parameters(model):,} trainable parameters")


    lgn_model = Seq2SeqLightning(
        model, lr=0.1, lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
        optimizer=torch.optim.SGD, loss_fn=torch.nn.CrossEntropyLoss, batch_size=BATCH_SIZE,
        momentum=0.9, weight_decay=1e-4
    )
    lr_monitor_callback = lgn.pytorch.callbacks.LearningRateMonitor(logging_interval="epoch")
    bar_callback = lgn.pytorch.callbacks.RichProgressBar(leave=True)
    timer_callback = lgn.pytorch.callbacks.Timer()

    torch.set_float32_matmul_precision('medium')  # For better performance with cuda

    lgn_trainer = Seq2SeqTrainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        precision="16-mixed",
        log_every_n_steps=len(train_dataloader),
        callbacks=[
            bar_callback,
            timer_callback,
            lr_monitor_callback,
        ],

        accumulate_grad_batches=5,
        enable_model_summary=True,
        )


    lgn_trainer.fit(model=lgn_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



