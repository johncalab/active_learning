import torch
from torch import nn
import torch.nn.functional as F

from john.convseq import ConvSeq

class WordVocab:
    def __init__(self,vectors,emb_dim=None):
        self.token_to_index = {}
        self.index_to_token = {}
        self.token_to_vector = {}

        self.pad = "<PAD>"
        self.add_token(self.pad, vectors[self.pad])
        self.unk = "<UNK>"
        self.add_token(self.unk, vectors[self.unk])
        self.bos = "<BOS>"
        self.add_token(self.bos, vectors[self.bos])
        self.eos = "<EOS>"
        self.add_token(self.eos, vectors[self.eos])

        for token in vectors.token_to_index:
            self.add_token(token, vectors[token])

    def __len__(self) -> int:
        return len(self.token_to_index)

    def add_token(self,token: str, vector):
        assert token not in self.token_to_index, f"{token} is already in vocab"
        index = len(self.token_to_index)
        self.token_to_index[token] = index
        self.index_to_token[index] = token
        self.token_to_vector[token] = vector

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


class WordEmbeddingPreTrained(nn.Module):
    def __init__(
        self,
        vocab,
    ):
        super().__init__()

        embeddings = [0] * len(vocab.token_to_index)
        for token in vocab.token_to_index:
            tensor = vocab.token_to_vector[token]
            idx = vocab.token_to_index[token]
            embeddings[idx] = tensor
        embeddings = torch.stack(embeddings)

        self.emb = nn.Embedding.from_pretrained(embeddings)

    def forward(self,x):
        return self.emb(x)

class WordEncoder(nn.Module):
    """
    Combine embeddings with CNN layers
    input: (batch_len,seq_len)
    output: (batch_len,seq_len,emb_dim)
    """
    def __init__(
        self,
        vocab,
        num_cnns,
        kernel_size,
        padding_idx=None,
        add_residual=True,
        dropout_p=0.0,
    ):
        super().__init__()

        self.emb = WordEmbeddingPreTrained(vocab)
        self.emb_dim = self.emb.emb.embedding_dim

        self.cnns = ConvSeq(
                in_dim=self.emb_dim,
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
        return x

if __name__ == "__main__":
    # example 0
    from torchnlp.word_to_vector import GloVe
    vectors = GloVe(cache=".word_vectors_cache")
    vocab = WordVocab(vectors)

    token = "fountain"
    idx1 = vocab.token_to_index[token]
    print(idx1)
    idx2 = vectors.token_to_index[token]
    print(idx2)

    v = vocab.token_to_vector["fountain"]
    w = vectors["fountain"]
    print(v-w)

    # example 1
    from torchnlp.word_to_vector import GloVe
    vectors = GloVe(cache=".word_vectors_cache")
    vocab = WordVocab(vectors)
    m = WordEmbeddingPreTrained(vocab=vocab)
    index = 25
    x = [index]
    x = torch.tensor(x)
    x = x.unsqueeze(dim=0)
    x = m(x)
    x = x.squeeze(dim=0)
    print(vocab.index_to_token[index])
    print(vectors.index_to_token[index])
    v = vectors[vocab.index_to_token[index]]
    print(x-v)

    # example 2
    from torchnlp.word_to_vector import GloVe
    vectors = GloVe(cache=".word_vectors_cache")
    vocab = WordVocab(vectors)
    m = WordEncoder(
        vocab = vocab,
        num_cnns = 2,
        kernel_size = 3,
    )

    index = 24
    x = [index]
    x = torch.tensor(x)
    x = x.unsqueeze(dim=0)
    print(x.size())
    x = m(x)
    print(x.size())




















