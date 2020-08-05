from typing import List, Dict
import torch
from torch.utils.data import Dataset

from bald.chars import CharVocab
from bald.words import WordVocab

class ConllDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        char_vocab: CharVocab,
        word_vocab: WordVocab,
    ):
        self.data = self.load_conll(data_path)
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab

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

    def load_conll(self, path: str,make_lower=False) -> List[Dict[str,List[str]]]:
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

    def __len__(self):
        return len(self.data)
        
    def word_to_char_ids(self, word: str) -> List[str]:
        return [self.char_vocab.get_index(c) for c in word]
        
    def pad_chars(self,list_of_seqs: List[List[int]]) -> List[List[int]]:
        seq_len = max([len(seq) for seq in list_of_seqs])
        pad_token = self.char_vocab.pad
        pad_index = self.char_vocab.get_index(pad_token)

        new_list = []
        for seq in list_of_seqs:
            rest = seq_len - len(seq)
            new_seq = seq[:] + [pad_index]*rest
            new_list.append(new_seq)

        return new_list
        
    def __getitem__(self,i: int) -> dict:
        sample = self.data[i]

        tag_seq = sample["tag"]
        tag_seq = [self.encoding[tag] for tag in tag_seq]
        word_seq = sample["text"]
        char_seq = [self.word_to_char_ids(w) for w in word_seq]
        word_seq = [self.word_vocab.get_index(w) for w in word_seq]

        return {"word":word_seq, "char":char_seq, "tag":tag_seq}

def collate_conll(batch: dict,word_pad: int,char_pad: int,tag_pad: int):
    """
    batch is a list of samples from the conll dataset
    each sample in batch is a dictionary with keys: word,char,tag
    
    word is List[int], a list of words
    tag is List[int], a list of tags
    char is List[List[int]], a list of words where each word is a list of chars
    
    Goal: pad everything so they have the same length
    
    Returns: dictionary of three tensors
    """
    max_char_len = max([len(word) for sent in batch for word in sent["char"] ])
    max_sent_len = max([len(sent["word"]) for sent in batch])
    
    def pad_tag(sent):
        remainder = max_sent_len - len(sent)
        return sent[:]+([tag_pad]*remainder)

    def pad_word(sent):
        remainder = max_sent_len - len(sent)
        return sent[:]+([word_pad]*remainder)
    
    def pad_char(sent):
        new_sent = []
        # pad each seq of chars
        for seq in sent:
            remainder = max_char_len - len(seq)
            new_sent.append(seq[:]+([char_pad]*remainder))
        
        # pad each sentence
        remainder = max_sent_len - len(sent)
        return new_sent[:]+([[char_pad]*max_char_len]*remainder)

    def pad(sent,key):
        if key == "tag":
            return pad_tag(sent[key])
        if key == "word":
            return pad_word(sent[key])
        if key == "char":
            return pad_char(sent[key])
   
    out = {}
    for key in ["tag","word","char"]:
        out[key] = [pad(sent,key) for sent in batch]
        out[key] = torch.tensor(out[key])
    
    return out

if __name__=="__main__":
    from bald import conll_path, vectors_path
    from torchnlp.word_to_vector import GloVe
    path = conll_path / "eng.testa"
    char_vocab = CharVocab()
    vectors = GloVe(cache=vectors_path)
    word_vocab = WordVocab(vectors=vectors)
    d = ConllDataset(data_path=path, char_vocab=char_vocab, word_vocab=word_vocab)
    it = d.__getitem__(3)
    for key in it:
        print(it[key].size())


