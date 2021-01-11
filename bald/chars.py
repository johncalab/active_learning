from string import printable
import torch
from torch import nn
import torch.nn.functional as F

from .convseq import ConvSeq
from .vocab import VocabBase


class CharVocab(VocabBase):
    """
    Vocabulary class for basic characters.
    Encoding is fixed, so no need to pickle dictionaries.
    0123456789
    abcdefghijklmnopqrstuvwxyz
    ABCDEFGHIJKLMNOPQRSTUVWXYZ
    !"#$%&'()*+,-./:;<=>?@[ \\ ]^_`{|}~
    """

    def __init__(self):
        super().__init__()

        for c in printable:
            self.add_token(c)


class CharEncoder(nn.Module):
    """
    Combine embeddings with CNN layers
    and reduce dimension with mean
    input: (batch_len,seq_len)
    output: (batch_len,emb_dim)

    idea: batch of sents if of shape
    (batch_len, max_sent_len)

    unpacking words as seqs of characters, we get
    (batch_len, max_sent_len, max_word_len)

    view it as
    (b*m, max_word_len)

    apply encoder to get
    (b*m, emb_dim)

    view it as
    (batch_len, max_sent_len, emb_dim)

    """

    def __init__(
        self,
        vocab_len: int,
        embedding_dim: int,
        num_cnns: int,
        kernel_size: int,
        padding_idx: int = None,
        add_residual: bool =True,
        dropout_p: float =0.0,
    ):
        super().__init__()

        self.emb = nn.Embedding(
            num_embeddings=vocab_len,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.cnns = ConvSeq(
            in_dim=embedding_dim,
            num_cnns=num_cnns,
            kernel_size=kernel_size,
            add_residual=add_residual,
            dropout_p=dropout_p,
        )

    def forward(self, x):
        x = self.emb(x)
        x = F.relu(x)
        x = self.cnns(x)
        x = F.relu(x)
        x = torch.mean(x, dim=1)
        return x
