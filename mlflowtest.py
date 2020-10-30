import datasets as hfds
from torchnlp.word_to_vector import GloVe
import mlflow
import torch
from tqdm import tqdm
from bald.vectorizers import ConllVectorizer
from bald.datasets import NERDataset
from bald.archs import BasicPaper
from bald.evals import ner_loss, conll_evaluate, conll_score
from bald.const import num_labels_with_pad

mlflow.set_experiment('BaldConll2003 Linear Decoder')
with mlflow.start_run() as run:

    # params
    model_params = {
        "num_cnns":3,
        "kernel_size":3,
    }

    params = {
        "batch_size":64,
        "loss_fun":"cross-entropy",
        "optimizer":"Adam default",
        "num_epochs":2,
    }
    for d in [model_params,params]:
        for param,val in d.items():
            mlflow.log_param(param,val)

    # setup
    vectors_path = '.vectors'
    data_path = '.data/conll'

    try:
        conll = hfds.load_from_disk(data_path)
    except:
        conll = hfds.load_dataset('conll2003')
        conll.save_to_disk(data_path)
        
    vectors = GloVe(cache=vectors_path)
    vzr = ConllVectorizer.from_vectors(vectors=vectors)

    model = BasicPaper.from_vectorizer(
        vzr=vzr,
        num_labels=num_labels_with_pad,
        **model_params
    )

    ds = {}
    dl = {}
    for key in ['train','validation','test']:
        ds[key] = NERDataset(data=conll[key],vzr=vzr)
        shuffle = False if key == 'test' else True
        dl[key] = ds[key].get_dataloader(batch_size=params["batch_size"],shuffle=shuffle)

    criterium = ner_loss # modify these two to fetch from param
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(1,params["num_epochs"]+1):
        print(f"Epoch {epoch}.\n")
        # training loop
        print("Train.")
        model.train()
        for datapoint in tqdm(dl['train']):
                
            y_true,c,w = datapoint['tag'],datapoint['char'],datapoint['word']
            y_pred = model(w=w,c=c)
            
            optimizer.zero_grad()
            loss = criterium(y_pred=y_pred,y_true=y_true)
            loss.backward()
            optimizer.step()
            mlflow.log_metric('train_loss',loss.item())
            
        # evaluation loop
        print("Valid.")
        model.eval()
        for datapoint in tqdm(dl['validation']):

            y_true,c,w = datapoint['tag'],datapoint['char'],datapoint['word']
            y_pred = model(w=w,c=c)
            
            loss = criterium(y_pred=y_pred,y_true=y_true)
            mlflow.log_metric('val_loss',loss.item())
    















