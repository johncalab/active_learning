from bald.encoders import CNNEncoder
from bald.decoders import LinearDecoder
from bald.chars import CharVocab
from bald.words import WordVocab

from torch import nn
import torch.nn.functional as F

class BasicPaper(nn.Module):
    """
    input: two tensors
        (b_len, seq_len)
        (b_len, seq_len, word_len)

    output: (b_len,seq_len,num_tags)
    """
    def __init__(
        self,
        char_vocab: CharVocab,
        word_vocab: WordVocab,
        num_tags: int,
        num_cnns: int,
        # char_num_cnns: int,
        # word_num_cnns: int,
        kernel_size: int,
        # char_kernel_size: int,
        # word_kernel_size: int,
        add_residual: bool=True,
        # char_add_residual: bool=True,
        # word_add_residual: bool=True,
        dropout_p: float = 0.0,
        # char_dropout_p: float=0.0,
        # word_dropout_p: float=0.0,
        # char_pad_idx: int = None,
        # word_pad_idx: int=None,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            char_vocab = char_vocab,
            word_vocab = word_vocab,
            num_cnns = num_cnns,
            char_num_cnns = num_cnns,
            word_num_cnns = num_cnns,
            kernel_size = kernel_size,
            char_kernel_size = kernel_size,
            word_kernel_size = kernel_size,
            add_residual = True,
            char_add_residual = True,
            word_add_residual = True,
            dropout_p = dropout_p,
            char_dropout_p = dropout_p,
            word_dropout_p = dropout_p,
            char_pad_idx = 0,
            word_pad_idx = 0,
        )

        in_dim = 2*self.encoder.emb_dim
        self.decoder = LinearDecoder(in_dim=in_dim,num_tags=num_tags)

    def forward(self,w,c):
        """w is word tensor, c is char tensor"""
        x = self.encoder(w,c)
        x = F.relu(x)
        return self.decoder(x)


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

    m = BasicPaper(
        word_vocab=word_vocab,
        char_vocab=char_vocab,
        num_cnns=2,
        kernel_size=3,
        num_tags=7,
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
        print('\n')