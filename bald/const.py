conll_tags = {
    'O':0,
    'B-PER':1,
    'I-PER':2,
    'B-ORG':3,
    'I-ORG':4,
    'B-LOC':5,
    'I-LOC':6,
    'B-MISC':7,
    'I-MISC':8,
}

conll_reverse_tags = {value:key for key,value in conll_tags.items()}

conll_encoding = {
    'O':5,
    'B-PER':1,
    'I-PER':1,
    'B-ORG':2,
    'I-ORG':2,
    'B-LOC':3,
    'I-LOC':3,
    'B-MISC':4,
    'I-MISC':4,
}

conll_decoding = {
    5:'O',
    1:'PER',
    2:'ORG',
    3:'LOC',
    4:'MISC',
}

num_labels = len(conll_decoding)
num_labels_with_pad = num_labels + 1

assert len(set(conll_encoding.values())) == num_labels, 'Mismatch.'
