from typing import List, Dict
import torch
from torch.utils.data import Dataset

from .vocab import NERTagVocab
from .chars import CharVocab
from .words import WordVocab, WordVocabScratch

class VectorizerBase:
    def __init__(self,vocab):
        self.vocab = vocab

    def vectorize_sequence(self,seq):
        return [self.vocab.get_index(tok) for tok in seq]

    def vectorize_batch(self,batch):
        return [self.vectorize_sequence(seq) for seq in batch]

    def vectorize_and_pad_batch(self,batch):
        max_len = max(len(seq) for seq in batch)        
        pad_idx = self.vocab.pad_idx
        unpadded = self.vectorize_batch(batch)
        padded = []
        for seq in unpadded:
            remainder = max_len - len(seq)
            padded.append(seq[:]+([pad_idx]*remainder))

        return padded

class WordVectorizer(VectorizerBase):
    @classmethod
    def from_corpus(cls,corpus):
        vocab = WordVocabScratch(corpus)
        return cls(vocab=vocab)

class CharVectorizer(VectorizerBase):

    def vectorize_sequence(self,seq):
        return [[self.vocab.get_index(c) for c in tok] for tok in seq]

    def vectorize_and_pad_batch(self,batch):
        max_len = max(len(char_seq) for seq in batch for char_seq in seq)
        pad_idx = self.vocab.pad_idx
        unpadded = self.vectorize_batch(batch)
        padded = []
        for seq in unpadded:
            padded_char_seqs = []
            for char_seq in seq:
                remainder = max_len - len(char_seq)
                padded_char_seqs.append(char_seq[:]+([pad_idx]*remainder))

            padded.append(padded_char_seqs)

        return padded


class NERTagVectorizer(VectorizerBase):
    def __init__(self,tag_encoding,tag_decoding):
        self.vocab = NERTagVocab(tag_encoding=tag_encoding,tag_decoding=tag_decoding)
        self.vocab.pad_idx = 0
