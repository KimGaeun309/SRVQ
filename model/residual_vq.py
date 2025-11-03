import torch
import torch.nn as nn
from typing import Sequence


class ReferenceEncoder(torch.nn.Module):
    """Reference encoder module.

    This module is reference encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.

    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017

    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the referece encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.

    """

    def __init__(
        self,
        idim=80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder, self).__init__()

        # check hyperparameters are valid
        assert conv_kernel_size % 2 == 1, "kernel size must be odd."
        assert (
            len(conv_chans_list) == conv_layers
        ), "the number of conv layers and length of channels list must be the same."

        convs = []
        padding = (conv_kernel_size - 1) // 2
        for i in range(conv_layers):
            conv_in_chans = 1 if i == 0 else conv_chans_list[i - 1]
            conv_out_chans = conv_chans_list[i]
            convs += [
                torch.nn.Conv2d(
                    conv_in_chans,
                    conv_out_chans,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    # Do not use bias due to the following batch norm
                    bias=False,
                ),
                torch.nn.BatchNorm2d(conv_out_chans),
                torch.nn.ReLU(inplace=True),
            ]
        self.convs = torch.nn.Sequential(*convs)

        self.conv_layers = conv_layers
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = padding

        # get the number of GRU input units
        gru_in_units = idim
        for i in range(conv_layers):
            gru_in_units = (
                gru_in_units - conv_kernel_size + 2 * padding
            ) // conv_stride + 1
        gru_in_units *= conv_out_chans
        self.gru = torch.nn.GRU(gru_in_units, gru_units, gru_layers, batch_first=True)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.

        Args:
            speech (Tensor): Batch of padded target features (B, Lmax, idim).

        Returns:
            Tensor: Reference embedding (B, gru_units)

        """
        batch_size = speech.size(0)
        xs = speech.unsqueeze(1)  # (B, 1, Lmax, idim)
        hs = self.convs(xs).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
        # NOTE(kan-bayashi): We need to care the length?
        time_length = hs.size(1)
        hs = hs.contiguous().view(batch_size, time_length, -1)  # (B, Lmax', gru_units)
        self.gru.flatten_parameters()
        _, ref_embs = self.gru(hs)  # (gru_layers, batch_size, gru_units)
        ref_embs = ref_embs[-1]  # (batch_size, gru_units)

        return ref_embs


class VectorQuantizer(torch.nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(
        self,
        n_e: int = 13,
        e_dim: int = 256,
        beta: float = 0.2
    ):
        super(VectorQuantizer, self).__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.vq_embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # z [B, H, W=80]
        z = z.unsqueeze(1) # [B, C, 256]
        z = z.permute(0, 2, 1).contiguous() # [B, 256, C] 
        z_flattened = z.view(-1, self.e_dim) # [?, 256]

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.vq_embedding.weight**2, dim=1) \
            - 2 * torch.matmul(z_flattened, self.vq_embedding.weight.t())
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.vq_embedding.weight).view(z.shape) # [B, 256, C] 

        # compute loss for embedding
        vq_loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach() # [B, 256, C] 

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q_out = z_q.permute(0, 1, 2).contiguous() # [B, 256, C] 

        z_q_out = z_q_out.squeeze(-1) # [B, 256]

        return z_q_out, vq_loss, min_encoding_indices


class ResidualVQ(torch.nn.Module):
    def __init__(
        self,
        idim: int = 80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        n_e: int = 13,
        e_dim: int = 256,
        num_vq: int = 3,
        beta: float = 0.2,
    ):
        super(ResidualVQ, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )

        # self.vq_layer_stack = nn.ModuleList(
        #     [
        #         VectorQuantizer(n_e=n_e, e_dim=e_dim) for _ in range(num_vq)
        #     ]
        # )
        
        self.vq_layer1 = VectorQuantizer(n_e=n_e, e_dim=e_dim)
        self.vq_layer2 = VectorQuantizer(n_e=n_e, e_dim=e_dim)
        self.vq_layer3 = VectorQuantizer(n_e=n_e, e_dim=e_dim)
        # self.vq_layer4 = VectorQuantizer(n_e=n_e, e_dim=e_dim)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        ref_embs = self.ref_enc(speech) # [16, H, W=80] -> [16, 256]

        residual = ref_embs
        z_q_out_1, vq_loss_1, min_encoding_indices_1 = self.vq_layer1(residual)

        residual = residual - z_q_out_1
        z_q_out_2, vq_loss_2, min_encoding_indices_2 = self.vq_layer2(residual)

        residual = residual - z_q_out_2
        z_q_out_3, vq_loss_3, min_encoding_indices_3 = self.vq_layer3(residual)

        # residual = residual - z_q_out_3
        # z_q_out_4, vq_loss_4, min_encoding_indices_4 = self.vq_layer4(residual)

        # vq_loss_total = 0
        # codebooks = []
        # residual = ref_embs
        # for i, vq_layer in enumerate(self.vq_layer_stack):
        #     z_q_out, vq_loss, min_encoding_indices = vq_layer(residual)
        #     if i == 0:
        #         z_q = torch.cat([z_q_out, z_q_out], dim=1)
        #     else:
        #         z_q = torch.cat([z_q, z_q_out], dim=1)
            
        #     codebooks.append(z_q_out)
        #     residual = residual - z_q_out
        #     vq_loss_total += vq_loss

        # # vq4
        # codebooks = [z_q_out_1, z_q_out_2, z_q_out_3, z_q_out_1, z_q_out_2, z_q_out_3]
        # z_q_out = torch.cat([z_q_out_1, z_q_out_2, z_q_out_3, z_q_out_4], dim=1)
        # vq_loss = vq_loss_1 + vq_loss_2 + vq_loss_3 + vq_loss_4

        # vq3
        codebooks = [z_q_out_1, z_q_out_2, z_q_out_3, z_q_out_1 + z_q_out_2 + z_q_out_3]
        z_q_out = torch.cat([z_q_out_1, z_q_out_2, z_q_out_3], dim=1) # [B=16, 256*3]
        vq_loss = vq_loss_1 + vq_loss_2 + vq_loss_3

        # # vq2
        # codebooks = [z_q_out_1, z_q_out_2, z_q_out_1, z_q_out_2]
        # z_q_out = torch.cat([z_q_out_1, z_q_out_2], dim=1)
        # vq_loss = vq_loss_1 + vq_loss_2

        return z_q_out, vq_loss, min_encoding_indices_1, codebooks


if __name__ == "__main__":
    ref_embs = torch.rand((8, 256))
    vq_layer = ResidualVQ(n_e=8, e_dim=256)

    vq_style_embs, vq_loss, min_encoding_indices = vq_layer(ref_embs)
    print(vq_style_embs.size())