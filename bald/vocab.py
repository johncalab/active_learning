class VocabBase:
    def __init__(self):
        self.token_to_index = {}
        self.index_to_token = {}

        self.pad = "<PAD>"
        self.pad_idx = self.add_token(self.pad)
        self.unk = "<UNK>"
        self.add_token(self.unk)
        self.bos = "<BOS>"
        self.add_token(self.bos)
        self.eos = "<EOS>"
        self.add_token(self.eos)

    def __len__(self) -> int:
        return len(self.token_to_index)

    def add_token(self, token: str) -> int:
        if token in self.token_to_index:
            index = self.token_to_index[token]
        else:
            index = len(self.token_to_index)
            self.token_to_index[token] = index
            self.index_to_token[index] = token
        return index

    def get_index(self, token: str) -> int:
        if token in self.token_to_index:
            return self.token_to_index[token]
        else:
            return self.token_to_index[self.unk]

    def get_token(self, j: int) -> str:
        if j in self.index_to_token.keys():
            return self.index_to_token[j]
        else:
            raise KeyError(f"{j} not a valid index.")

class NERTagVocab(VocabBase):
    def __init__(self,tag_encoding,tag_decoding):
        self.token_to_index = tag_encoding
        self.index_to_token = tag_decoding
        self.pad_idx = 0

    def add_token(self):
        raise NotImplementedError("This method is disabled for this subclass.")
