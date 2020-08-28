import torch
import torch.nn.functional as F
from bald.const import conll_decoding

def conll_evaluate(seq,vzr,model):
    v = vzr.vectorize(seq)
    c = torch.tensor(v["char"]).unsqueeze(dim=0)
    w = torch.tensor(v["word"]).unsqueeze(dim=0)
    x = model(w=w,c=c)
    x = F.softmax(x, dim=2)
    x = torch.argmax(x,dim=2)
    x = x.squeeze(dim=0)
    return [conll_decoding[a.item()] for a in x]
