import torch

from bald.datasets import ConllDataset, conll_collate_batch
from bald.vectorizers import WordVectorizer, CharVectorizer, NERTagVectorizer
from bald.chars import CharVocab
from bald.const import conll_encoding, conll_decoding

def test_conllds():
    data = [
        {'words':["I","location","you"],'ner':['O','I-LOC','B-MISC']},
        {'words':["You","verb","something","else"],'ner':['O','O','O','B-ORG']},
        {'words':["Hello","goodbye"],'ner':['I-PER','B-MISC']},
    ]
    corpus = ["I love bananas", "you love bananas", "I hate hoomans"]
    corpus = [sent.split() for sent in corpus]

    char_vzr = CharVectorizer(CharVocab())
    word_vzr = WordVectorizer.from_corpus(corpus)
    tag_vzr = NERTagVectorizer(
        tag_encoding=conll_encoding,
        tag_decoding=conll_decoding,
    )

    ds = ConllDataset(data=data)

    assert len(ds) == 3
    assert {
        'word':['I','location','you'],
        'tag':['O','I-LOC','B-MISC'],
    } == ds.__getitem__(0)

def test_conllds_batch():
    data = [
        {'words':["I","location","you"],'ner':['O','I-LOC','B-MISC']},
        {'words':["You","verb","something","else"],'ner':['O','O','O','B-ORG']},
        {'words':["Hello","goodbye"],'ner':['I-PER','B-MISC']},
    ]
    corpus = ["I love bananas", "you love bananas", "I hate hoomans"]
    corpus = [sent.split() for sent in corpus]

    char_vzr = CharVectorizer(CharVocab())
    word_vzr = WordVectorizer.from_corpus(corpus)
    tag_vzr = NERTagVectorizer(
        tag_encoding=conll_encoding,
        tag_decoding=conll_decoding,
    )

    ds = ConllDataset(data=data)
    batch = [ds.__getitem__(i) for i in range(3)]
    batch = conll_collate_batch(
        batch,
        char_vzr=char_vzr,
        word_vzr=word_vzr,
        tag_vzr=tag_vzr,
    )

    word_vectorized = [
        [word_vzr.vocab.get_index(w) for w in ['I','location','you','<PAD>']],
        [word_vzr.vocab.get_index(w) for w in ['You','verb','something','else']],
        [word_vzr.vocab.get_index(w) for w in ['Hello','goodbye','<PAD>','<PAD>']],
    ]
    assert word_vectorized == batch['word']

    tag_vectorized = [
        [tag_vzr.vocab.get_index(t) for t in ['O','I-LOC','B-MISC']] + [0],
        [tag_vzr.vocab.get_index(t) for t in ['O','O','O','B-ORG']],
        [tag_vzr.vocab.get_index(t) for t in ['I-PER','B-MISC']] + [0]*2,
    ]
    assert tag_vectorized == batch['tag']

    char_vectorized = [
        [
            [char_vzr.vocab.get_index(c) for c in ['I','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['l','o','c','a','t','i','o','n','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['y','o','u','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
    ],
        [
            [char_vzr.vocab.get_index(c) for c in ['Y','o','u','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['v','e','r','b','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['s','o','m','e','t','h','i','n','g']],
            [char_vzr.vocab.get_index(c) for c in ['e','l','s','e','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
    ],
        [
            [char_vzr.vocab.get_index(c) for c in ['H','e','l','l','o','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['g','o','o','d','b','y','e','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
            [char_vzr.vocab.get_index(c) for c in ['<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>','<PAD>']],
        ],
    ]
    assert char_vectorized == batch['char']

def some_torch_test(): 
    """
    word_sequence ---> tensor ---> model ---> interpretation of output
    same with a batch

    do directly batch, and treat word as batch of one!
    """
    vocab = CharVocab()
    vzr = CharVectorizer(vocab)
    seq = ['I', 'want', 'to', 'go', 'home']
    batch = [seq]
    batch = vzr.vectorize_and_pad_batch(batch)
    x = torch.tensor(batch)
    a,b,c = x.size()
    assert a == 1
    assert b == 5
    assert c == 4
















