from typing import List, Dict
import torch
from torch.utils.data import Dataset

from bald.chars import CharVocab
from bald.words import WordVocab

class CharVectorizer:

    def __init__(self,vocab:CharVocab):
        self.vocab = vocab

    def word_to_char_ids(self, word: str) -> List[str]:
        return [self.vocab.get_index(c) for c in word]

    def vectorize(self,word_seq: List[str]) -> List[int]:
        """
        word_seq is a list of strings
        (so already tokenized)
        """
        return [self.word_to_char_ids(w) for w in word_seq]

    def vectorize_pad(self,sent):
        char_seq = self.vectorize(sent)
        char_pad = self.vocab.pad_idx
        max_len = max(len(seq) for seq in char_seq)
        new_seq = []
        for seq in char_seq:
            remainder = max_len - len(seq)
            new_seq.append(seq[:]+([char_pad]*remainder))
        return new_seq

class WordVectorizer:

    def __init__(self,vocab:WordVocab):
        self.vocab = vocab
        
    def vectorize(self,word_seq:List[str])->List[int]:
        return [self.vocab.get_index(w) for w in word_seq]

class ConllVectorizer:

    def __init__(self,char_vocab,word_vocab):
        self.char_vzr = CharVectorizer(char_vocab)
        self.word_vzr = WordVectorizer(word_vocab)

    def vectorize(self,seq):
        c = self.char_vzr.vectorize_pad(seq)
        w = self.word_vzr.vectorize(seq)
        return {"char":c,"word":w}
