import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from .const import conll_decoding, num_labels

def ner_loss(y_pred,y_true):
    batch_len,seq_len,num_labels = y_pred.size()
    y_pred = y_pred.view(batch_len*seq_len,num_labels)
    y_true = y_true.view(batch_len*seq_len)
    return F.cross_entropy(
        input=y_pred,
        target=y_true,
        ignore_index=0,
        weight = torch.tensor([0,1,1,1,1,0.1])
    )

def conll_evaluate(seq,vzr,model):
    decoding = conll_decoding.copy()
    decoding[0] = '<PAD>'
    v = vzr.vectorize(seq)
    c = torch.tensor(v["char"]).unsqueeze(dim=0)
    w = torch.tensor(v["word"]).unsqueeze(dim=0)
    x = model(w=w,c=c)
    x = F.softmax(x, dim=2)
    x = torch.argmax(x,dim=2)
    x = x.squeeze(dim=0)
    return [decoding[a.item()] for a in x]

def conll_score(y_pred,y_true):
    batch_len,seq_len,num_labels = y_pred.size()
    y_pred = y_pred.view(batch_len*seq_len,num_labels)
    y_pred = F.softmax(y_pred,dim=1)
    y_pred = torch.argmax(y_pred,dim=1)
    y_true = y_true.view(batch_len*seq_len)
    return f1_score(
        y_true = y_true.cpu().data.numpy(),
        y_pred = y_pred.cpu().data.numpy(),
        labels = list(range(1,num_labels+1)),
        average = 'micro',
    )
