from typing import Any, Tuple
import torch
from torch import nn
import random
import tqdm
import lightning as lgn
from lightning import Callback

from torchtext.transforms import VocabTransform, ToTensor, PadTransform, Sequential, RegexTokenizer, AddToken

def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float | torch.Tensor:
    true_classes = torch.argmax(y_true, dim=1)
    pred_classes = torch.argmax(torch.sigmoid(y_pred), dim=1)
    # num of hits / num of classes mean over the batch
    return (true_classes == pred_classes).sum() / y_pred.size(0)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        return hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        bidirectional_mult = 2 if encoder.bidirectional else 1
        assert (
            (encoder.n_layers * bidirectional_mult == decoder.n_layers)
        ), "Encoder and decoder must have equal number of layers (or 2times if encoder is bidirectional)!"

    def forward(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)

        input = trg[0, :]
        for t in range(1, trg_length):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
        return outputs


class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, src, trg, src_vocab, trg_vocab, pad_token="<pad>", sos_token = "<sos>", eos_token = "<eos>"):
        self.src, self.trg = src, trg

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.pad_token, self.sos_token, self.eos_token = pad_token, sos_token, eos_token

        self.src_ids, self.trg_ids, self.max_length, self.src_transform, self.trg_transform = None, None, None, None, None
        self._setup()

    def _setup(self):
        self.max_len = max(
            max(map(len, self.src)),
            max(map(len, self.trg)) 
            ) + 3 # for safe padding

        self.src_transform = Sequential(
            VocabTransform(self.src_vocab),
            ToTensor(padding_value=self.src_vocab[self.pad_token]),
            PadTransform(pad_value=self.src_vocab[self.pad_token], max_length=self.max_len)
        )
        self.trg_transform = Sequential(
            VocabTransform(self.trg_vocab),
            ToTensor(padding_value=self.trg_vocab[self.pad_token]),
            PadTransform(pad_value=self.trg_vocab[self.pad_token], max_length=self.max_len)
        )

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]
    
    def collate_fn(self, batch):
        src, trg = zip(*batch)
        src, trg = list(src), list(trg) # to list to avoid error
        src = self.src_transform(src)
        trg = self.trg_transform(trg)
        return src, trg
    
class Seq2SeqLightning(lgn.LightningModule):
    def __init__(self, model, batch_size, lr, lr_scheduler, optimizer, loss_fn, accuracy_fn=accuracy, momentum=None, weight_decay=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.loss_fn = loss_fn()
        self.accuracy_fn = accuracy_fn
        self.batch_size = batch_size

        self.teacher_forcing_ratio = 0.5

        self.momentum = momentum if momentum is not None else 0
        self.weight_decay = weight_decay if weight_decay is not None else 0
    
    def forward(self, src, trg):
        return self.model(src, trg, self.teacher_forcing_ratio)
    
    def configure_optimizers(self):
        if self.optimizer == torch.optim.SGD:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.lr_scheduler is not None:
            if self.lr_scheduler == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, mode="min", factor=0.05, patience=5, cooldown=5, min_lr=1e-6)
            elif self.lr_scheduler == torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
                self.lr_scheduler = self.lr_scheduler(self.optimizer, T_0 = 5, T_mult=2, eta_min=1e-6)
            else:
                raise NotImplementedError(f"Scheduler {self.lr_scheduler} not implemented")      
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler, 'monitor': 'train_loss'}
            
    def _base_step(self, batch) -> Tuple[Any, Any]:
        src, trg = batch
        out = self.model(src, trg, self.teacher_forcing_ratio)
        out = out[1:].view(-1, out.shape[-1])
        trg = trg[1:].view(-1)

        loss = self.loss_fn(out, trg)
        with torch.no_grad():
            # acc = self.accuracy_fn(out, trg)
            acc = 0.4242
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=True, on_epoch=False, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"val_loss": loss, "val_acc": acc}, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._base_step(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc})


class Seq2SeqTrainer(lgn.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, model, teacher_forcing_ratio=0.5, train_dataloaders=None, val_dataloaders=None, datamodule=None,
            ckpt_path=None):
        model.teacher_forcing_ratio = teacher_forcing_ratio
        super().fit(model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders, datamodule=datamodule,
                     ckpt_path=ckpt_path)