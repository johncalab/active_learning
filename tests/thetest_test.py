"""
* load data
* load model
* feed data to model
* interpret output
* given new sentence, feed it to model
"""
import torch
from bald.datasets import ConllDataset, conll_collate_batch
from bald.vectorizers import WordVectorizer, CharVectorizer, NERTagVectorizer
from bald.chars import CharVocab
from bald.words import WordVocabScratch
from bald.const import conll_encoding, conll_decoding

def _test():
    
    char_vocab = CharVocab()
    word_vocab = WordVocabScratch(corpus=corpus)
