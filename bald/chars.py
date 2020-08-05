from string import printable
import torch
from torch import nn
import torch.nn.functional as F

from bald.convseq import ConvSeq

class CharVocab:
    """
    Vocabulary class for basic characters.
    Encoding is fixed, so no need to pickle dictionaries.
    0123456789
    abcdefghijklmnopqrstuvwxyz
    ABCDEFGHIJKLMNOPQRSTUVWXYZ
    !"#$%&'()*+,-./:;<=>?@[ \\ ]^_`{|}~ 
    """
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}

        self.pad = "<PAD>"
        self.add_token(self.pad)
        self.unk = "<UNK>"
        self.add_token(self.unk)
        self.bos = "<BOS>"
        self.add_token(self.bos)
        self.eos = "<EOS>"
        self.add_token(self.eos)

        for c in printable:
            self.add_token(c)

    def __len__(self) -> int:
        return len(self.token_to_index)

    def add_token(self,token: str) -> int:
        if token in self.token_to_index:
            index = self.token_to_index[token]
        else:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return index

    def get_index(self,token: str) -> int:
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return self.token_to_index[self.unk]

    def get_token(self,j: int) -> str:
        if j in self.index_to_token.keys():
            return self.index_to_token[j]
        else:
            raise KeyError(f"{j} not a valid index.")


class CharEmbedding(nn.Module):
    """
    input: (batch_len,seq_len)
    output: (batch_len,seq_len,emb_dim)
    """
    def __init__(
        self,
        vocab_len,
        embedding_dim,
        padding_idx=None,
        ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings = vocab_len,
            embedding_dim = embedding_dim,
            padding_idx = padding_idx,
            )

    def forward(self,x):
        x = self.embedding(x)
        return x


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
        vocab_len,
        embedding_dim,
        num_cnns,
        kernel_size,
        padding_idx=None,
        add_residual=True,
        dropout_p=0.0,
    ):
        super().__init__()

        self.emb = CharEmbedding(
                vocab_len=vocab_len,
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

    def forward(self,x):
        x = self.emb(x)
        x = F.relu(x)
        x = self.cnns(x)
        x = F.relu(x)
        x = torch.mean(x,dim=1)
        return x

if __name__=="__main__":
    vocab = CharVocab()

    m = CharEncoder(
        vocab_len = len(vocab),
        embedding_dim = 300,
        num_cnns = 2,
        kernel_size = 3,
        )

    word = [c for c in "hello."]
    word = [vocab.token_to_index[c] for c in word]
    word = [torch.tensor(c) for c in word]
    x = torch.stack(word)
    x = x.unsqueeze(dim=0)
    print(x.size())
    y = m(x)
    print(y.size())












