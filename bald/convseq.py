import torch
from torch import nn

class ConvSeq(nn.Module):
    """
    input: (batch_len, seq_len, in_dim)

    apply CNN layers with fixed config
    and optional additive residual connections
    to obtain

    output: (batch_len, seq_len, out_dim)
    """
    def __init__(
        self,
        in_dim: int,
        num_cnns: int,
        kernel_size: int,
        add_residual: bool=True,
        dropout_p: float=0.0,
        ):
        """
        We have stride 1, dilation 1.
        To maintain dims, padding should be
        p = (k-1)/2

        So kernel size should be odd.
        """
        assert isinstance(kernel_size,int)
        assert kernel_size > 0
        assert kernel_size % 2 == 1
        super().__init__()
        self.add_residual = add_residual
        padding_size = int((kernel_size - 1) / 2)

        cnn_kwargs = {
            "in_channels":1,
            "out_channels":in_dim,
            "kernel_size":(kernel_size,in_dim),
            "padding":(padding_size,0),
        }
        conv_layers = []
        for _ in range(num_cnns):
            conv_layers.append(nn.Sequential(
                    nn.Conv2d(**cnn_kwargs),
                    nn.Dropout(dropout_p),
                    nn.ReLU(),
                ))
        self.convs = nn.ModuleList(conv_layers)

    def forward(self,x):
        # x has shape (batch_len,seq_len,in_dim)
        # add channel dim (batch_len,1,seq_len,in_dim)
        x = x.unsqueeze(dim=1)
        # after each CNN layer, output will have shape
        # (batch_len, in_dim, seq_len, 1)
        x_old = x
        for conv in self.convs:
            x = conv(x)
            x = torch.transpose(x,1,3)
            if self.add_residual:
                x = x + x_old
                x_old = x

        # x has now shape (batch_len,1,seq_len,in_dim)
        x = x.squeeze(dim=1)
        return x
