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
        raise NotImplementedError

    def vectorize_batch(self,batch):
        raise NotImplementedError

    def vectorize_word(self,word: str):
        return [self.vocab.get_index(c) for c in word]

    def vectorize_and_pad_word(self,word,total_len: int):
        p = self.vectorize_word(word)
        remainder = total_len - len(p)
        assert remainder >= 0, "total_len is too small"
        pad_idx = self.vocab.pad_idx
        p = p[:] + [pad_idx]*remainder
        return p

    def vectorize_and_pad_word_seq(self,word_seq,total_char_len,total_word_len):
        padded = []
        for word in word_seq:
            char_seq = self.vectorize_and_pad_word(word=word,total_len=total_char_len)
            padded.append(char_seq)

        remainder = total_word_len - len(padded)
        assert remainder >= 0, "total_word_len is too small"
        empty_word = [self.vocab.pad_idx]*total_char_len
        for i in range(remainder):
            padded.append(empty_word)
        return padded

    def vectorize_and_pad_batch(self,batch):
        total_word_len = max(len(seq) for seq in batch)
        total_char_len = max(len(char_seq) for seq in batch for char_seq in seq)

        padded = []
        for seq in batch:
            new_seq = self.vectorize_and_pad_word_seq(
                word_seq=seq,
                total_char_len=total_char_len,
                total_word_len=total_word_len,
            )
            padded.append(new_seq)
        return padded

        # batch_padded = []
        # for seq in batch:
        #     seq_padded = 



        # pad_idx = self.vocab.pad_idx
        # unpadded = self.vectorize_batch(batch)
        # padded = []
        # for seq in unpadded:
        #     padded_char_seqs = []
        #     for char_seq in seq:
        #         remainder = max_len - len(char_seq)
        #         padded_char_seqs.append(char_seq[:]+([pad_idx]*remainder))

        #     padded.append(padded_char_seqs)

        # return padded


class NERTagVectorizer(VectorizerBase):
    def __init__(self,tag_encoding,tag_decoding):
        self.vocab = NERTagVocab(tag_encoding=tag_encoding,tag_decoding=tag_decoding)
