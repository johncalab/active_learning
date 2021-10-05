import datasets as hfds
from .const import conll_reverse_tags

def load_hf_data(path: str, dataset_name: str):
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

def convert_conll_tags(seq):
    """seq is List[str] of tags"""
    return [conll_reverse_tags[tag] for tag in seq]

def convert_from_hf(hfdata):
    out = []
    for i in range(len(hfdata)):
        datapoint = hfdata[i]
        new = {}
        new['words'] = datapoint['tokens']
        new['ner'] = convert_conll_tags(datapoint['ner_tags'])
        out.append(new)
    return out
