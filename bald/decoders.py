from torch import nn

class LinearDecoder(nn.Module):
    """
    input
    (b_len, seq_len, in_dim)

    output
    (b_len, seq_len, num_labels)
    """
    def __init__(
        self,
        in_dim,
        num_labels,
    ):
        super().__init__()

        self.fc = nn.Linear(in_dim,num_labels)

    def forward(self,x):
        return self.fc(x)
