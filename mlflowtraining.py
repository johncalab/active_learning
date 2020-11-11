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


def log_this(vzr,model):
    seq = ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.','<PAD>','<PAD>']
    ev = conll_evaluate(seq,vzr=vzr,model=model)
    print(ev)

def average(array):
    if len(array) == 0:
        return 0

    return sum(array)/len(array)

def load_data(path: str, dataset_name: str):
    """
    Looks for data in path.
    If it finds it, it loads it.
    If not, it uses huggingface to download it from the interwebs.
    Returns the hugginface dataset.
    """

    try:
        conll = hfds.load_from_disk(path)
    except:
        conll = hfds.load_dataset(dataset_name)
        conll.save_to_disk(path)

    return conll

def train_model(model, dataloader, optimizer, criterium, score_fun):
    model.train()
    losses = []
    scores = []
    for datapoint in tqdm(dataloader):
            
        y_true,c,w = datapoint['tag'],datapoint['char'],datapoint['word']
        y_pred = model(w=w,c=c)
        
        optimizer.zero_grad()
        loss = criterium(y_pred=y_pred,y_true=y_true)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        score = score_fun(y_pred=y_pred,y_true=y_true)
        scores.append(score)

    return {
        'score':average(scores),
        'loss':average(losses),
    }

def evaluate_model(model, dataloader, criterium, score_fun):
    model.eval()
    losses = []
    scores = []
    for datapoint in tqdm(dataloader):

        y_true,c,w = datapoint['tag'],datapoint['char'],datapoint['word']
        y_pred = model(w=w,c=c)
        
        loss = criterium(y_pred=y_pred,y_true=y_true)
        losses.append(loss.item())
        score = score_fun(y_pred=y_pred,y_true=y_true)
        scores.append(score)

    return {
        'score':average(scores),
        'loss':average(losses)
    }


if __name__=='__main__':

    # params
    # TODO: add loss and optimizer to params
    model_params = {
        "num_cnns":5,
        "kernel_size":5,
    }
    params = {
        "batch_size":64,
        "dataset_name":"conll2003",
        "num_epochs":3,
    }

    mlflow.set_experiment('BaldConll2003 Linear Decoder')
    with mlflow.start_run() as run:

        # log params
        for d in [model_params,params]:
            for param,val in d.items():
                mlflow.log_param(param,val)

        # load data
        data_path = '.data/conll'
        conll = load_data(path=data_path,dataset_name=params['dataset_name'])

        # load vectors
        vectors_path = '.vectors'
        vectors = GloVe(cache=vectors_path)

        # make vectorizer
        vzr = ConllVectorizer.from_vectors(vectors=vectors)

        # create model
        model = BasicPaper.from_vectorizer(
            vzr=vzr,
            num_labels=num_labels_with_pad,
            **model_params
        )

        # create datasets and dataloaders
        ds = {}
        dl = {}
        for key in ['train','validation','test']:
            ds[key] = NERDataset(data=conll[key],vzr=vzr)
            # shuffle = False if key == 'test' else True
            shuffle = False
            dl[key] = ds[key].get_dataloader(batch_size=params["batch_size"],shuffle=shuffle)

        # set loss function and optimizer
        criterium = ner_loss
        optimizer = torch.optim.Adam(model.parameters())

        # save best score/loss/model and evaluate on test set?
        for epoch in range(1,params["num_epochs"]+1):
            print(f"Epoch {epoch}.\n")

            print("Train")
            # train epoch and return loss and score on training set
            result = train_model(
                model=model,
                dataloader=dl['train'],
                optimizer=optimizer,
                criterium=criterium,
                score_fun=conll_score,
            )
            mlflow.log_metric('train_loss',result['loss'],step=epoch)
            mlflow.log_metric('train_score',result['score'],step=epoch)

            print("Valid.")
            # compute loss and score on validtion set
            result = evaluate_model(
                model=model,
                dataloader=dl['validation'],
                criterium=criterium,
                score_fun=conll_score,
            )
            mlflow.log_metric('valid_loss',result['loss'],step=epoch)
            mlflow.log_metric('valid_score',result['score'],step=epoch)
