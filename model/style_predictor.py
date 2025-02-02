import torch
import numpy as np
from torch.nn import functional as F


class StylePredictor(torch.nn.Module):
    """Mel style encoder"""
    def __init__(
        self,
        idim=256,
        style_hidden=256,
        style_vector_dim=256,
        style_kernel_size=5,
        style_head=2,
        dropout=0.1,
    ):
        """Initilize mel style encoder module"""
        super(StylePredictor, self).__init__()

        self.in_dim = idim
        self.hidden_dim = style_hidden
        self.out_dim = style_vector_dim
        self.kernel_size = style_kernel_size
        self.n_head = style_head
        self.dropout = dropout

        self.spectral = torch.nn.Sequential(
            LinearNorm(self.in_dim, self.hidden_dim),
            Mish(),
            torch.nn.Dropout(self.dropout),
            LinearNorm(self.hidden_dim, self.hidden_dim),
            Mish(),
            torch.nn.Dropout(self.dropout)
        )

        self.temporal = torch.nn.Sequential(
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
            Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size, self.dropout),
        )

        self.slf_attn = MultiHeadAttention(self.n_head, self.hidden_dim, 
                                self.hidden_dim//self.n_head, self.hidden_dim//self.n_head, self.dropout) 

        self.fc = LinearNorm(self.hidden_dim, self.out_dim)

    def temporal_avg_pool(self, x, mask=None):
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def forward(self, x, mask=None):
        max_len = x.shape[1]
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1) if mask is not None else None

        # spectral
        x = self.spectral(x)
        # temporal
        x = x.transpose(1,2)
        x = self.temporal(x)
        x = x.transpose(1,2)
        # self-attention
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0)
        x, _ = self.slf_attn(x, mask=slf_attn_mask)
        # fc
        x = self.fc(x)
        # temoral average pooling
        w = self.temporal_avg_pool(x, mask=mask)

        return w


class LinearNorm(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            bias=True, 
            spectral_norm=False,
        ):
        super(LinearNorm, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels, bias)
        
        if spectral_norm:
            self.fc = torch.nn.utils.spectral_norm(self.fc)

    def forward(self, input):
        out = self.fc(input)
        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dGLU(torch.nn.Module):
    '''
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.
    '''
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            dropout
        ):
        super(Conv1dGLU, self).__init__()
        self.out_channels = out_channels
        self.conv1 = ConvNorm(in_channels, 2*out_channels, kernel_size=kernel_size)
        self.dropout = torch.nn.Dropout(dropout)
            
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True, 
            spectral_norm=False,
        ):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)
        
        if spectral_norm:
            self.conv = torch.nn.utils.spectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


class MultiHeadAttention(torch.nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0., spectral_norm=False):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = torch.nn.Linear(d_model, n_head * d_k)
        self.w_ks = torch.nn.Linear(d_model, n_head * d_k)
        self.w_vs = torch.nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(temperature=np.power(d_model, 0.5), dropout=dropout)

        self.fc = torch.nn.Linear(n_head * d_v, d_model)
        self.dropout = torch.nn.Dropout(dropout)

        if spectral_norm:
            self.w_qs = torch.nn.utils.spectral_norm(self.w_qs)
            self.w_ks = torch.nn.utils.spectral_norm(self.w_ks)
            self.w_vs = torch.nn.utils.spectral_norm(self.w_vs)
            self.fc = torch.nn.utils.spectral_norm(self.fc)

    def forward(self, x, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_x, _ = x.size()

        residual = x

        q = self.w_qs(x).view(sz_b, len_x, n_head, d_k)
        k = self.w_ks(x).view(sz_b, len_x, n_head, d_k)
        v = self.w_vs(x).view(sz_b, len_x, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_x, d_v)  # (n*b) x lv x dv

        if mask is not None:
            slf_mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        else:
            slf_mask = None
        output, attn = self.attention(q, k, v, mask=slf_mask)

        output = output.view(n_head, sz_b, len_x, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
                        sz_b, len_x, -1)  # b x lq x (n*dv)

        output = self.fc(output)

        output = self.dropout(output) + residual
        return output, attn


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, dropout):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        p_attn = self.dropout(attn)

        output = torch.bmm(p_attn, v)
        return output, attn


class StylePredictor2(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            filter_channels=256,
            kernel_size=3,
            p_dropout=0.5,
            gin_channels=0,
    ):
        """Initilize style predictor module."""
        super(StylePredictor2, self).__init__()
        
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = torch.nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = torch.nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x)
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)
