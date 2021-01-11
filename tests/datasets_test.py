from bald.datasets import ConllDataset

def test_conllds():
    words = [
        ["I","location","you"],
        ["You","verb","something","else"],
        ["Hello","goodbye"],
    ]
    ner = [
        ['O','I-LOC','B-MISC'],
        ['O','O','O','B-ORG'],
        ['I-PER','B-MISC'],
    ]
    data = {'words':words,'ner':ner}
    corpus = ["I love bananas", "you love bananas", "I hate hoomans"]
    corpus = [sent.split() for sent in corpus]
