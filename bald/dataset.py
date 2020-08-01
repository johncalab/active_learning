from typing import List, Dict
import torch
from torch.utils.data import Dataset

from .chars import CharVocab
from .words import WordVocab

def load_conll(path: str,make_lower=False) -> List[Dict[str]]:
    """
    Read the conll dataset and make sense of it.

    Returns: a list of sentences
        each sentence is a python dictionary that looks like this
        {
            "text":["I", "live", "in", "Colorado"],
            "tag": ["O","O","O","LOC"],
        }
    """

    def parse_single_line(line):
        split = line.split()
        return (split[0], split[-1])

    with open(path, "r") as f:
        new_line = f.readline()

        sentences = []
        while new_line:

            if new_line == "\n":
                new_line = f.readline()
                continue

            new_sentence = {"text":[], "tag":[]}
            while new_line and new_line != "\n":
                text, tag = parse_single_line(new_line)
                if make_lower is True:
                    text = text.lower()
                new_sentence["text"].append(text)
                new_sentence["tag"].append(tag)
                new_line = f.readline()

            sentences.append(new_sentence)
            new_line = f.readline()

    return sentences


class ConllDataset(Dataset):
    def __init__(
        self,
        data_path: str, 
        char_vocab: CharVocab, 
        word_vocab: WordVocab,
    ):
        self.data = load_ner_dataset(data_path,make_lower=False)

        self.encoding = {
            'O':0,
            'B-PER':1,
            'I-PER':1,
            'B-ORG':2,
            'I-ORG':2,
            'B-LOC':3,
            'I-LOC':3,
            'B-MISC':4,
            'I-MISC':4,
        }
        self.num_labels = len(set(self.encoding.values()))

        self.max_seq_len = self.compute_max_seq_len()
        self.vectors = vectors
        self.emb_dim = emb_dim

    def compute_max_seq_len(self):
        return max(len(d["tag"]) for d in self.data)

    def set_max_seq_len(self,val: int):
        assert val > 0
        self.max_seq_len = val

    def word_to_char_ids(self, word: str) -> List[str]:
        id_seq = [self.char_vocab.token_to_index[c] for c in word]
        return torch.tensor(id_seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        sample = self.data[i]
        word_seq = sample["text"]
        tag_seq = sample["tag"]
        char_seq = [self.word_to_char_ids(w) for w in word_seq]




        x_seq = sample["text"]
        y_seq = sample["tag"]

        x_seq = [self.vectors[tok] for tok in x_seq]
        rest = self.max_seq_len - len(x_seq)
        assert rest >= 0
        x_seq.extend([torch.zeros(self.emb_dim) for _ in range(rest)])
        x_seq = torch.stack(x_seq)

        y_seq = [self.encoding[tok] for tok in y_seq]
        y_seq.extend([0 for _ in range(rest)])
        assert len(y_seq) == self.max_seq_len
        y_seq = torch.tensor(y_seq)

        return x_seq,y_seq
