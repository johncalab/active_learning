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
    assert False
