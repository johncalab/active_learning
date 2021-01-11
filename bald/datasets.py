from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

from .chars import CharVocab
from .words import WordVocab
from .const import conll_encoding

class ConllDataset(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i: int) -> dict:
        sample = self.data[i]
        return {'word':sample['words'], 'tag':sample['ner']}

def conll_collate_batch(batch,char_vzr,word_vzr,tag_vzr):
    word_batch = []
    tag_batch = []
    for element in batch:
        word_batch.append(element['word'])
        tag_batch.append(element['tag'])

    char_batch = char_vzr.vectorize_and_pad_batch(word_batch)
    word_batch = word_vzr.vectorize_and_pad_batch(word_batch)

    return {'char':char_batch,'word':word_batch}
