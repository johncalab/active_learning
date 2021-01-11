import pytest
import torch

from bald.words import WordVocabScratch, WordEncoderScratch

@pytest.fixture
def the_corpus():
    corpus = ["I love bananas", "you love bananas", "I hate hoomans"]
    return [sent.split() for sent in corpus]

def test_vocab(the_corpus):
    vocab = WordVocabScratch(the_corpus)
    assert len(vocab) == 6+4

def test_encoder(the_corpus):
    vocab = WordVocabScratch(the_corpus)
    encoder = WordEncoderScratch(
        vocab = vocab,
        embedding_dim = 10,
        num_cnns = 3,
        kernel_size = 3,
        add_residual = True,
        dropout_p = 0.3,
    )

    x = [
        [0,3,5,3,8],
        [1,2,3,4,7],
        [0,8,6,2,1],
    ]
    x = torch.tensor(x)
    y = encoder(x)

    assert tuple(y.size()) == (3,5,10)
