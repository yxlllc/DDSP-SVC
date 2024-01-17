import torch
from torch import nn


# From https://github.com/CNChTu/Diffusion-SVC/ by CNChTu
# License: MIT


class ConformerNaiveEncoder(nn.Module):
    """
    Conformer Naive Encoder

    Args:
        dim_model (int): Dimension of model
        num_layers (int): Number of layers
        num_heads (int): Number of heads
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.
        atten_dropout (float): Dropout rate of attention module, default 0.
    """

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 dim_model: int,
                 expansion_factor: int = 2,
                 kernel_size: int = 31,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.,
                 use_pre_norm=False,
                 conv_model_type='mode1'
                 ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.use_norm = use_norm
        self.residual_dropout = 0.1  # 废弃代码,仅做兼容性保留
        self.attention_dropout = 0.1  # 废弃代码,仅做兼容性保留

        self.encoder_layers = nn.ModuleList(
            [
                CFNEncoderLayer(
                    dim_model=dim_model,
                    expansion_factor=expansion_factor,
                    kernel_size=kernel_size,
                    num_heads=num_heads,
                    use_norm=use_norm,
                    conv_only=conv_only,
                    conv_dropout=conv_dropout,
                    atten_dropout=atten_dropout,
                    use_pre_norm=use_pre_norm,
                    conv_model_type=conv_model_type
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """

        for (i, layer) in enumerate(self.encoder_layers):
            x = layer(x, mask)
        return x  # (#batch, length, dim_model)


class CFNEncoderLayer(nn.Module):
    """
    Conformer Naive Encoder Layer

    Args:
        dim_model (int): Dimension of model
        expansion_factor (int): Expansion factor of conv module, default 2
        kernel_size (int): Kernel size of conv module, default 31
        num_heads (int): Number of heads
        use_norm (bool): Whether to use norm for FastAttention, only True can use bf16/fp16, default False
        conv_only (bool): Whether to use only conv module without attention, default False
        conv_dropout (float): Dropout rate of conv module, default 0.1
        atten_dropout (float): Dropout rate of attention module, default 0.1
    """

    def __init__(self,
                 dim_model: int,
                 expansion_factor: int = 2,
                 kernel_size: int = 31,
                 num_heads: int = 8,
                 use_norm: bool = False,
                 conv_only: bool = False,
                 conv_dropout: float = 0.,
                 atten_dropout: float = 0.1,
                 use_pre_norm=False,
                 conv_model_type='mode1'
                 ):
        super().__init__()
        self.use_pre_norm = use_pre_norm

        self.conformer = ConformerConvModule(
            dim_model,
            expansion_factor=expansion_factor,
            kernel_size=kernel_size,
            use_norm=use_norm,
            dropout=conv_dropout,
            use_pre_norm=use_pre_norm,
            conv_model_type=conv_model_type)

        self.norm = nn.LayerNorm(dim_model)

        self.dropout = nn.Dropout(0.1)  # 废弃代码,仅做兼容性保留

        # selfatt -> fastatt: performer!
        if not conv_only:
            self.attn = nn.TransformerEncoderLayer(
                d_model=dim_model,
                nhead=num_heads,
                dim_feedforward=dim_model * 4,
                dropout=atten_dropout,
                activation='gelu'
            )
        else:
            self.attn = None

    def forward(self, x, mask=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (#batch, length, dim_model)
            mask (torch.Tensor): Mask tensor, default None
        return:
            torch.Tensor: Output tensor (#batch, length, dim_model)
        """
        if self.attn is not None:
            if self.use_pre_norm:
                x = self.norm(x) + x
            else:
                x = self.norm(x)
            x = x + (self.attn(x, mask=mask))

        x = x + (self.conformer(x))

        return x  # (#batch, length, dim_model)


class ConformerConvModule(nn.Module):
    def __init__(
            self,
            dim,
            expansion_factor=2,
            kernel_size=31,
            dropout=0.,
            use_norm=False,
            conv_model_type='mode1',
            use_pre_norm=False
    ):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size)

        if conv_model_type == 'mode1':
            if use_norm:
                if use_pre_norm:
                    _norm = PreNorm(dim)
                else:
                    _norm = nn.LayerNorm(dim)
            else:
                _norm = nn.Identity()
            self.net = nn.Sequential(
                _norm,
                Transpose((1, 2)),
                nn.Conv1d(dim, inner_dim * 2, 1),
                nn.GLU(dim=1),
                nn.Conv1d(inner_dim, inner_dim, kernel_size=kernel_size, padding=padding[0], groups=inner_dim),
                nn.SiLU(),
                nn.Conv1d(inner_dim, dim, 1),
                Transpose((1, 2)),
                nn.Dropout(dropout)
            )
        elif conv_model_type == 'mode2':
            raise NotImplementedError('mode2 not implemented yet')
        else:
            raise ValueError(f'{conv_model_type} is not a valid conv_model_type')

    def forward(self, x):
        return self.net(x)


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class PreNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x) + x
