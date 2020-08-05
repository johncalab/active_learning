from string import printable
import torch
from torch import nn
import torch.nn.functional as F

from bald.convseq import ConvSeq
from bald.chars import CharVocab, CharEncoder
from bald.words import WordVocab, WordEncoder

class CNNEncoder(nn.Module):
    """
    Stack Char and Word encoders
    input: two tensors
        (b_len, seq_len)
        (b_len, seq_len, word_len)

    output: (b_len,seq_len,out_dim)
    """
    def __init__(
        self,
        char_vocab: CharVocab,
        word_vocab: WordVocab,
        num_cnns: int,
        char_num_cnns: int,
        word_num_cnns: int,
        kernel_size: int,
        char_kernel_size: int,
        word_kernel_size: int,
        add_residual: bool=True,
        char_add_residual: bool=True,
        word_add_residual: bool=True,
        dropout_p: float = 0.0,
        char_dropout_p: float=0.0,
        word_dropout_p: float=0.0,
        char_pad_idx: int = None,
        word_pad_idx: int=None,
    ):
        super().__init__()

        self.word_encoder = WordEncoder(
            vocab=word_vocab,
            num_cnns=word_num_cnns,
            kernel_size=word_kernel_size,
            padding_idx=word_pad_idx,
            add_residual=word_add_residual,
            dropout_p=word_dropout_p,
        )
        self.emb_dim = self.word_encoder.emb_dim

        self.char_encoder = CharEncoder(
            vocab_len = len(char_vocab),
            embedding_dim = self.emb_dim,
            num_cnns = char_num_cnns,
            kernel_size = char_kernel_size,
            padding_idx=char_pad_idx,
            add_residual=char_add_residual,
            dropout_p=char_dropout_p,
        )

        self.cnns = ConvSeq(
            in_dim=2*self.emb_dim,
            num_cnns=num_cnns,
            kernel_size=kernel_size,
            add_residual=add_residual,
            dropout_p=dropout_p,
        )


    def forward(self,w,c):
        """w is word tensor, c is char tensor"""
        w = self.word_encoder(w)

        batch_len,seq_len,word_len = c.size()
        c = c.view(batch_len*seq_len,word_len)
        c = self.char_encoder(c)
        c = c.view(batch_len,seq_len,-1)

        x = torch.cat([w,c],dim=2)
        x = F.relu(x)
        x = self.cnns(x)

        return x

if __name__=="__main__":
    from bald.datasets import ConllDataset, collate_conll
    from bald.words import WordVocab, WordEncoder
    from bald.chars import CharVocab, CharEncoder
    from bald import vectors_path, conll_path

    import torch
    from torch.utils.data import DataLoader
    from torchnlp.word_to_vector import GloVe

    data_path = conll_path / "eng.testa"
    char_vocab = CharVocab()
    vectors = GloVe(cache=vectors_path)
    word_vocab = WordVocab(vectors=vectors)
    ds = ConllDataset(data_path=data_path, char_vocab=char_vocab, word_vocab=word_vocab)

    dl = DataLoader(
        dataset=ds, 
        batch_size=4, 
        shuffle=True, 
        collate_fn = (lambda batch: collate_conll(batch=batch,word_pad=0,char_pad=0,tag_pad=0))
    )

    m = CNNEncoder(
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        num_cnns=2,
        word_num_cnns=2,
        char_num_cnns=2,
        kernel_size=3,
        word_kernel_size=3,
        char_kernel_size=3,
    )

    for i,s in enumerate(dl):
        if i > 10:
            break

        w = s["word"]
        print(w.size())
        c = s["char"]
        print(c.size())

        x = m(w=w,c=c)
        print(x.size())