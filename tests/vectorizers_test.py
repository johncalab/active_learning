import pytest

from bald.vectorizers import CharVectorizer, WordVectorizer, NERTagVectorizer
from bald.chars import CharVocab
from bald.words import WordVocabScratch
from bald.const import conll_encoding, conll_decoding

@pytest.fixture
def corpus():
    seq = [
        "I have money.",
        "You have nintendo.",
        "Twitter finally banned him.",
    ]
    return [sent.split() for sent in seq]

def test_word(corpus):
    vzr = WordVectorizer.from_corpus(corpus)

    seq = ["I","have","money"]
    vectorized = [vzr.vocab.get_index(tok) for tok in seq]

    assert vectorized == vzr.vectorize_sequence(seq)

def test_word_batch(corpus):
    vocab = WordVocabScratch(corpus)
    vzr = WordVectorizer(vocab)

    seq_1 = ["I","have","money"]
    seq_2 = ["hello"]
    batch = [seq_1,seq_2]
    seq_2_pad = ["hello","<PAD>","<PAD>"]
    vectorized = []
    for seq in [seq_1,seq_2_pad]:
        vectorized.append([vocab.get_index(tok) for tok in seq])

    assert vectorized == vzr.vectorize_and_pad_batch(batch)

def test_char():
    vocab = CharVocab()
    vzr = CharVectorizer(vocab)

    seq = ["I","have","money"]
    vectorized = [
        [vocab.get_index(c) for c in ['I','<PAD>','<PAD>','<PAD>','<PAD>']],
        [vocab.get_index(c) for c in ['h','a','v','e','<PAD>']],
        [vocab.get_index(c) for c in ['m','o','n','e','y']],
    ]
    assert vectorized == vzr.vectorize_and_pad_batch([seq])[0]

def test_char_batch(corpus):
    vocab = CharVocab()
    vzr = CharVectorizer(vocab)

    seq_1 = ["I","have","money"]
    seq_2 = ["helping"]
    batch = [seq_1,seq_2]

    char_seq_1 = [
        ['I','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>'],
        ['h','a','v','e','<PAD>','<PAD>','<PAD>'],
        ['m','o','n','e','y','<PAD>','<PAD>'],
    ]
    char_seq_2 = [
        ['h','e','l','p','i','n','g'],
        ['<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>'],
        ['<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>'],
    ]
    vectorized = [[[vocab.get_index(c) for c in c_seq] for c_seq in seq] for seq in [char_seq_1,char_seq_2]]

    assert vectorized == vzr.vectorize_and_pad_batch(batch)

def test_tag():
    vzr = NERTagVectorizer(tag_encoding=conll_encoding,tag_decoding=conll_decoding)
    seq = ['O','O','B-PER','I-PER','I-MISC','I-LOC']
    vectorized = [5,5,1,1,4,3]

    assert vectorized == vzr.vectorize_sequence(seq)

def test_tag_batch():
    vzr = NERTagVectorizer(tag_encoding=conll_encoding,tag_decoding=conll_decoding)
    seq_1 = ['O','O','B-PER','I-PER','I-MISC','I-LOC']
    seq_2 = ['O','I-MISC','O']
    batch = [seq_1,seq_2]
    vectorized = [
        [5,5,1,1,4,3],
        [5,4,5,0,0,0],
    ]

    assert vectorized == vzr.vectorize_and_pad_batch(batch)
