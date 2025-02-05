import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence
import pyworld as pw
import numpy as np
import librosa


class ReferenceEncoder(torch.nn.Module): # Original RefEnc
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

class ReferenceEncoderDynamic(torch.nn.Module):
    """Modified ReferenceEncoder for dynamically sized 1D vector inputs."""

    def __init__(
        self,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64,  128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,  # Set GRU output size to 128
    ):
        super(ReferenceEncoderDynamic, self).__init__()
        self.conv_layers = conv_layers
        self.conv_chans_list = conv_chans_list
        self.kernel_size = conv_kernel_size
        self.stride = conv_stride
        self.padding = (conv_kernel_size - 1) // 2
        self.gru_layers = gru_layers
        self.gru_units = gru_units

        # Conv layers
        convs = []
        for i in range(conv_layers):
            in_ch = 1 if i == 0 else conv_chans_list[i - 1]
            out_ch = conv_chans_list[i]
            convs.append(
                torch.nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=conv_kernel_size,
                    stride=conv_stride,
                    padding=self.padding,
                    bias=False,
                )
            )
            convs.append(torch.nn.BatchNorm1d(out_ch))
            convs.append(torch.nn.ReLU(inplace=True))
        self.convs = torch.nn.Sequential(*convs)

        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=self.conv_chans_list[-1],  # Match Conv1D output channels
            hidden_size=self.gru_units,          # Set GRU output size to 128
            num_layers=self.gru_layers,
            batch_first=True,
        )

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        """Forward propagation for 1D inputs."""
        batch_size, seq_len = input_vector.shape
        input_vector = input_vector.unsqueeze(1)  # Add channel dim: [B, 1, N]
        hs = self.convs(input_vector)  # [B, out_ch, reduced_seq_len]

        # print("hs after Conv1D:", hs.size())

        # Reshape for GRU
        hs = hs.permute(0, 2, 1)  # [B, reduced_seq_len, out_ch]
        self.gru.flatten_parameters()

        # print("hs for GRU:", hs.size())

        _, ref_embs = self.gru(hs)  # [num_layers, B, gru_units]

        # print("ref_embs:", ref_embs.size())
        return ref_embs[-1]  # Return last layer's output: [B, gru_units]




# import torch
# import torch.nn as nn
# from typing import Sequence

# class ReferenceEncoder(torch.nn.Module):
#     """Reference encoder module with low and high frequency band processing.

#     This module processes the mel-spectrogram by splitting it into low and high
#     frequency bands and applying separate CNNs for each band.

#     Args:
#         idim (int, optional): Dimension of the input mel-spectrogram.
#         conv_layers (int, optional): The number of conv layers in the reference encoder.
#         conv_chans_list: (Sequence[int], optional):
#             List of the number of channels of conv layers in the reference encoder.
#         conv_kernel_size (int, optional):
#             Kernel size of conv layers in the reference encoder.
#         conv_stride (int, optional):
#             Stride size of conv layers in the reference encoder.
#         gru_layers (int, optional): The number of GRU layers in the reference encoder.
#         gru_units (int, optional): The number of GRU units in the reference encoder.

#     """

#     def __init__(
#         self,
#         idim=80,
#         conv_layers: int = 6,
#         conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
#         conv_kernel_size: int = 3,
#         conv_stride: int = 2,
#         gru_layers: int = 1,
#         gru_units: int = 128,
#     ):
#         """Initilize reference encoder module."""
#         super(ReferenceEncoder, self).__init__()

#         # Conv layers for low frequency
#         self.conv_low_freq = self._build_conv_layers(
#             conv_layers, conv_chans_list, conv_kernel_size, conv_stride
#         )

#         # Conv layers for high frequency
#         self.conv_high_freq = self._build_conv_layers(
#             conv_layers, conv_chans_list, conv_kernel_size, conv_stride
#         )

#         # Calculate GRU input size after convolution
#         low_freq_gru_in_units = self._calc_gru_input_size(idim // 2, conv_kernel_size, conv_stride, conv_layers, conv_chans_list[-1])
#         high_freq_gru_in_units = low_freq_gru_in_units  # Same as low since we split evenly

#         # GRU layers
#         self.gru = nn.GRU(low_freq_gru_in_units + high_freq_gru_in_units, gru_units, gru_layers, batch_first=True)

#     def _build_conv_layers(self, conv_layers, conv_chans_list, conv_kernel_size, conv_stride):
#         """Helper function to create the CNN layers."""
#         layers = []
#         padding = (conv_kernel_size - 1) // 2
#         for i in range(conv_layers):
#             in_chans = 1 if i == 0 else conv_chans_list[i - 1]
#             out_chans = conv_chans_list[i]
#             layers += [
#                 nn.Conv2d(
#                     in_chans, out_chans,
#                     kernel_size=conv_kernel_size,
#                     stride=conv_stride,
#                     padding=padding,
#                     bias=False
#                 ),
#                 nn.BatchNorm2d(out_chans),
#                 nn.ReLU(inplace=True)
#             ]
#         return nn.Sequential(*layers)

#     def _calc_gru_input_size(self, idim, conv_kernel_size, conv_stride, conv_layers, out_channels):
#         """Helper function to calculate the input size for the GRU after convolution.""" 
#         gru_in_units = idim
#         padding = (conv_kernel_size - 1) // 2
#         for _ in range(conv_layers):
#             gru_in_units = (gru_in_units - conv_kernel_size + 2 * padding) // conv_stride + 1
#         return gru_in_units * out_channels

#     def forward(self, speech: torch.Tensor) -> torch.Tensor:
#         """Calculate forward propagation.

#         Args:
#             speech (Tensor): Batch of padded target features (B, Lmax, idim).

#         Returns:
#             Tensor: Reference embedding (B, gru_units)

#         """
#         batch_size = speech.size(0)
        
#         # Split the mel-spectrogram into low and high frequency bands
#         low_freq = speech[:, :, :40]  # Low frequency part (first 40 bins)
#         high_freq = speech[:, :, 40:]  # High frequency part (last 40 bins)

#         # Process low and high frequency bands separately
#         low_freq = low_freq.unsqueeze(1)  # (B, 1, Lmax, 40)
#         high_freq = high_freq.unsqueeze(1)  # (B, 1, Lmax, 40)
        
#         low_freq_out = self.conv_low_freq(low_freq).transpose(1, 2)  # (B, Lmax', conv_out_chans, idim')
#         high_freq_out = self.conv_high_freq(high_freq).transpose(1, 2)  # Same shape as low_freq_out

#         # Flatten both low and high frequency outputs
#         low_freq_out = low_freq_out.contiguous().view(batch_size, low_freq_out.size(1), -1)  # (B, Lmax', low_freq_features)
#         high_freq_out = high_freq_out.contiguous().view(batch_size, high_freq_out.size(1), -1)  # (B, Lmax', high_freq_features)

#         # Concatenate low and high frequency outputs
#         combined_out = torch.cat([low_freq_out, high_freq_out], dim=2)  # (B, Lmax', low_freq_features + high_freq_features)

#         # Pass through GRU
#         self.gru.flatten_parameters()
#         _, ref_embs = self.gru(combined_out)  # (gru_layers, batch_size, gru_units)
#         ref_embs = ref_embs[-1]  # (batch_size, gru_units)

#         return ref_embs

class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_classes),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.classifier(x)

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
        n_e: int = 7,
        e_dim: int = 256,
        beta: float = 0.4
    ):
        super(VectorQuantizer, self).__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.vq_embedding = torch.nn.Embedding(self.n_e, self.e_dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        # self.vq_embedding.weight.data.normal_(mean=0.0, std=1.0)

        

        

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

        z = z.unsqueeze(1)
        z = z.permute(0, 2, 1).contiguous() 
        z_flattened = z.view(-1, self.e_dim) 


        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.vq_embedding.weight**2, dim=1) \
            - 2 * torch.matmul(z_flattened, self.vq_embedding.weight.t())
        
        
        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        print("min_encoding_indices", min_encoding_indices)

        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.vq_embedding.weight).view(z.shape) # [B, 256, C] 

        # compute loss for embedding
        vq_loss = torch.mean((z_q.detach() - z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)
        
        # diversity_loss = torch.mean(torch.cdist(self.vq_embedding.weight, self.vq_embedding.weight, p=2)**2)
        # vq_loss += diversity_loss * 0.01
        
        # print("vq_loss", vq_loss)

        # print("vq_loss 1", torch.mean((z_q.detach() - z)**2) )
        # print("vq_loss 2", self.beta , "*",  torch.mean((z_q - z.detach()) ** 2))
        # print("vq_loss", vq_loss)
        # print("==============================================================")
        
        # preserve gradients
        z_q = z + (z_q - z).detach() # [B, 256, C] 

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        
        usage_loss = -torch.sum(e_mean * torch.log(e_mean + 1e-10))

        vq_loss += 0.01 * usage_loss

        # reshape back to match original input shape
        z_q_out = z_q.permute(0, 1, 2).contiguous() # [B, 256, C] 

        z_q_out = z_q_out.squeeze(-1) # [B, 256] 250103 수정 

        return z_q_out, vq_loss, min_encoding_indices, perplexity

class VectorQuantizer_reset(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE with random restart.
    After every epoch, run:
    random_restart()
    reset_usage()
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    - usage_threshold : codes below threshold will be reset to a random code
    """

    def __init__(self, n_e=1024, e_dim=256, beta=0.25, usage_threshold=1.0e-9):
        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.usage_threshold = usage_threshold

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        
        # initialize usage buffer for each code as fully utilized
        self.register_buffer('usage', torch.ones(self.n_e), persistent=False)
        
        self.perplexity = None
        self.loss = None

    def dequantize(self, z):
        z_flattened = z.view(-1, self.e_dim)
        z_q = self.embedding(z_flattened).view(z.shape)
        return z_q

    def update_usage(self, min_enc):
        self.usage[min_enc] = self.usage[min_enc] + 1  # if code is used add 1 to usage
        self.usage /= 2 # decay all codes usage

    def reset_usage(self):
        self.usage.zero_() #  reset usage between epochs

    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        rand_codes = torch.randperm(self.n_e)[0:len(dead_codes)]
        with torch.no_grad():
            self.embedding.weight[dead_codes] = self.embedding.weight[rand_codes]

    def forward(self, z, return_indices=False):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.unsqueeze(1)
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).type_as(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = z_q.view(z.shape)

        self.update_usage(min_encoding_indices)

        # compute loss for embedding
        self.loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
                    torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        self.perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 1, 2).contiguous()
        z_q = z_q.squeeze(-1)
        
        return z_q, self.loss, min_encoding_indices

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
        n_e: int = 7,
        e_dim: int = 256,
        num_vq: int = 3,
        beta: float = 0.4,
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
        
        self.vq_layer1 = VectorQuantizer_reset(n_e=n_e, e_dim=e_dim)
        self.vq_layer2 = VectorQuantizer_reset(n_e=n_e, e_dim=e_dim)
        self.vq_layer3 = VectorQuantizer_reset(n_e=n_e, e_dim=e_dim)
        # self.vq_layer4 = VectorQuantizer(n_e=n_e, e_dim=e_dim)

    def forward(self, speech: torch.Tensor) -> torch.Tensor:
        ref_embs = self.ref_enc(speech) # [16, H, W=80] -> [16, 256]

        # ref_embs = speech # 3SRVQ 위해

        print("ref_embs", ref_embs.shape)

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

class ResidualVQ2(torch.nn.Module):
    def __init__(
        self,
        e_dim: int = 128,  # Each VQ outputs 128-dim vectors
        n_e: int = 7,
        num_vq: int = 2,  # Use only 2 VQ layers
        beta: float = 0.4,
    ):
        super(ResidualVQ2, self).__init__()
        self.vq_layer1 = VectorQuantizer(n_e=n_e, e_dim=e_dim)
        self.vq_layer2 = VectorQuantizer(n_e=n_e, e_dim=e_dim)

    def forward(self, input_vector: torch.Tensor):
        residual = input_vector
        z_q_out_1, vq_loss_1, min_encoding_indices_1, perplexity_1 = self.vq_layer1(residual)

        residual = residual - z_q_out_1
        z_q_out_2, vq_loss_2, min_encoding_indices_2, perplexity_2 = self.vq_layer2(residual)
        z_q_out = torch.cat([z_q_out_1, z_q_out_2], dim=1)  # [B, 128 + 128]
        vq_loss = vq_loss_1 + vq_loss_2
        codebooks = [z_q_out_1, z_q_out_2, z_q_out]

        perplexity = [perplexity_1, perplexity_2]
        return z_q_out, vq_loss, min_encoding_indices_1, codebooks, perplexity


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3)
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=5, dilation=5)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.lrelu(out)
        out = self.conv3(out)
        return out + residual


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling along time dimension
        self.gmp = nn.AdaptiveMaxPool1d(1)  # Global Max Pooling along time dimension
        
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels * 2, in_channels, kernel_size=1, bias=False),  # Concatenated so 2*in_channels
            nn.LeakyReLU(),
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        gap = self.gap(x)  # (B, C, 1)
        gmp = self.gmp(x)  # (B, C, 1)
        combined = torch.cat([gap, gmp], dim=1)  # Concatenate along channel dimension (B, 2*C, 1)
        out = self.fc(combined)  # (B, C, 1)
        return out.expand_as(x)  # Expand to match the input shape (B, C, sequence_length)


class TimeAttention(nn.Module):
    def __init__(self, in_channels):
        super(TimeAttention, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool1d(1)  # Global Average Pooling for time dimension
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.LeakyReLU(),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        gap = self.gap(x).squeeze(-1)  # Global Average Pooling -> (B, C)
        out = self.fc(gap)  # Apply FC layers -> (B, C)
        return out.unsqueeze(-1).expand_as(x)  # Expand back to (B, C, sequence_length)


class DualAttention(nn.Module):
    def __init__(self, in_channels):
        super(DualAttention, self).__init__()
        self.residual_block = ResidualBlock(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.time_attention = TimeAttention(in_channels)

    def forward(self, F: torch.Tensor):
        # 1. Apply residual block to get F'
        F = F.unsqueeze(1)  # Add channel dimension, input shape becomes (B, 1, sequence_length)
        F_prime = self.residual_block(F)  # Residual Block output shape: (B, 1, sequence_length)
        
        # 2. Channel attention
        W_c = self.channel_attention(F_prime)  # Channel attention output shape: (B, 1, sequence_length)
        
        # 3. Time attention
        W_t = self.time_attention(F_prime)  # Time attention output shape: (B, 1, sequence_length)
        
        # 4. Apply element-wise multiplication: F' * W_c * W_t
        out = F_prime * W_c * W_t
        
        return out.squeeze(1)  # Return to shape (B, sequence_length)

class SRVQ2(torch.nn.Module):
    def __init__(
        self,
        idim: int = 80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
        n_e: int = 7,
        e_dim: int = 256,
        num_vq: int = 3,
        beta: float = 0.4,
    ):
        super(SRVQ2, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )

        self.RVQ1 = ResidualVQ(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
            n_e=n_e,
            e_dim=e_dim // 2,
            num_vq=num_vq,
        )

        self.RVQ2 = ResidualVQ(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
            n_e=n_e,
            e_dim=e_dim // 2,
            num_vq=num_vq,
        )

        self.dual_attention = DualAttention(in_channels=1)



    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        ref_embs = self.ref_enc(ref_embs) # [16, H, W=80] -> [16, 256]
        ref_embs = self.dual_attention(ref_embs)

        z_low, z_high = torch.split(ref_embs, 128, dim=1)
        z_r = z_high - z_low

        style_ref_embs_1, vq_loss_1, min_encoding_indices_1, codebooks_1 = self.RVQ1(z_low)
        style_ref_embs_2, vq_loss_2, min_encoding_indices_2, codebooks_2 = self.RVQ2(z_r) 

        

        # style_ref_embs torch.Size([16, 384])
        # vq_loss torch.Size([])
        # min_encoding_indices  torch.Size([16, 1])



        # style_ref_embs = torch.concat(style_ref_embs_1, (style_ref_embs_1 + style_ref_embs_2))
        style_ref_embs = torch.cat([style_ref_embs_1, style_ref_embs_1 + style_ref_embs_2], dim=1)

        vq_loss = vq_loss_1 + vq_loss_2
        min_encoding_indices = min_encoding_indices_1 + min_encoding_indices_2
        codebooks = [torch.cat([cb1, cb1 + cb2], dim=1) for cb1, cb2 in zip(codebooks_1, codebooks_2)]

        return style_ref_embs, vq_loss, min_encoding_indices, codebooks


class SRVQ3(torch.nn.Module):
    def __init__(
        self,
        idim: int = 80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 64,
        n_e: int = 7,
        e_dim: int = 128,
        num_vq: int = 3,
        beta: float = 0.4,
    ):
        super(SRVQ3, self).__init__()

        self.ref_encp = ReferenceEncoderDynamic()

        self.ref_encd = ReferenceEncoderDynamic()

        self.ref_ence = ReferenceEncoderDynamic()

        self.RVQp = ResidualVQ2()

        self.RVQd = ResidualVQ2()

        self.RVQe = ResidualVQ2()


        self.dual_attention = DualAttention(in_channels=1)



    def forward(self, ref_embs: torch.Tensor, p_targets: torch.Tensor, d_targets: torch.Tensor, e_targets: torch.Tensor) -> torch.Tensor:
        # ref_embs = self.ref_enc(ref_embs) # [16, H, W=80] -> [16, 256]
        # ref_embs = self.dual_attention(ref_embs)

        # print("p_targets", p_targets.size())

        z_pitch = self.ref_encp(p_targets.float())

        z_duration = self.ref_encd(d_targets.float())

        z_energy = self.ref_ence(e_targets.float())

        # print("z_pitch", z_pitch.shape)
        # print("z_duration", z_duration.shape)
        # print("z_energy", z_energy.shape)

        style_ref_embs_1, vq_loss_1, min_encoding_indices_1, codebooks_1, perplexity_1 = self.RVQp(z_pitch)
        style_ref_embs_2, vq_loss_2, min_encoding_indices_2, codebooks_2, perplexity_2 = self.RVQd(z_duration)
        style_ref_embs_3, vq_loss_3, min_encoding_indices_3, codebooks_3, perplexity_3 = self.RVQe(z_energy) 

        # print("perplexity", perplexity_1, perplexity_2, perplexity_3)
        

        # style_ref_embs torch.Size([16, 384])
        # vq_loss torch.Size([])
        # min_encoding_indices  torch.Size([16, 1])



        # style_ref_embs = torch.concat(style_ref_embs_1, (style_ref_embs_1 + style_ref_embs_2))
        style_ref_embs = torch.cat([style_ref_embs_1, style_ref_embs_2, style_ref_embs_3], dim=1)

        vq_loss = vq_loss_1 + vq_loss_2 + vq_loss_3
        min_encoding_indices = min_encoding_indices_1 + min_encoding_indices_2 + min_encoding_indices_3

        

        # codebooks = [torch.cat([cb1, cb2, cb3], dim=1) for cb1, cb2, cb3 in zip(codebooks_1, codebooks_2, codebooks_3)]


        # print("codebooks_1", len(codebooks_1), codebooks_1[0].size())

        # print("style_ref_embs", style_ref_embs.size())

        # codebooks = codebooks_1 + codebooks_2 + codebooks_3
    
        # print("codebooks", len(codebooks))

        codebooks = [style_ref_embs_1, style_ref_embs_2, style_ref_embs_3, style_ref_embs_1+style_ref_embs_2+style_ref_embs_3]


        return style_ref_embs, vq_loss, min_encoding_indices, codebooks


class SRVQPyworld(torch.nn.Module):
    def __init__(
        self,
        idim: int = 80,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 64,
        n_e: int = 7,
        e_dim: int = 128,
        num_vq: int = 3,
        beta: float = 0.4,
    ):
        super(SRVQPyworld, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )

        self.RVQp = ResidualVQ2()
        self.RVQd = ResidualVQ2()
        self.RVQe = ResidualVQ2()


    # ref_embs : original mel-spectrogram , pitch_mels : pitch normalized mel-spectrogram, energy_mels : energy normalized mel-spectrogram
    def forward(self, ref_embs: torch.Tensor, pitch_mels: torch.Tensor, energy_mels: torch.Tensor) -> torch.Tensor:
        # Convert ref_embs to numpy array for processing
        

        # Pass through encoders and RVQs
        ref_encoded = self.ref_enc(ref_embs)
        pitch_encoded = self.ref_enc(pitch_mels)
        energy_encoded = self.ref_enc(energy_mels)

        style_ref_embs_1, vq_loss_1, min_encoding_indices_1, codebooks_1, perplexity_1 = self.RVQp(ref_encoded)
        style_ref_embs_2, vq_loss_2, min_encoding_indices_2, codebooks_2, perplexity_2 = self.RVQe(pitch_encoded)
        style_ref_embs_3, vq_loss_3, min_encoding_indices_3, codebooks_3, perplexity_3 = self.RVQd(energy_encoded)

        # Combine style embeddings
        style_ref_embs = torch.cat([style_ref_embs_1, style_ref_embs_2, style_ref_embs_3], dim=1)

        # Combine VQ losses
        vq_loss = vq_loss_1 + vq_loss_2 + vq_loss_3
        min_encoding_indices = [min_encoding_indices_1, min_encoding_indices_2, min_encoding_indices_3]
        codebooks = [style_ref_embs_1, style_ref_embs_2, style_ref_embs_3]

        return style_ref_embs, vq_loss, min_encoding_indices, codebooks




if __name__ == "__main__":
    ref_embs = torch.rand((8, 256))
    vq_layer = ResidualVQ(n_e=8, e_dim=256)

    vq_style_embs, vq_loss, min_encoding_indices = vq_layer(ref_embs)
    print(vq_style_embs.size())
