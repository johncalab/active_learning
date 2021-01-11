from bald.chars import CharVocab, CharEncoder
import torch

def test_chars():
    """simple test to make sure things at least run"""
    vocab = CharVocab()

    m = CharEncoder(
        vocab_len=len(vocab),
        embedding_dim=300,
        num_cnns=2,
        kernel_size=3,
    )

    word = [c for c in "hello."]
    word = [vocab.token_to_index[c] for c in word]
    word = [torch.tensor(c) for c in word]
    x = torch.stack(word)
    x = x.unsqueeze(dim=0)
    print(x.size())
    y = m(x)
    print(y.size())

    assert True
