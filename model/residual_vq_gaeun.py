import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import KMeans



from typing import Sequence

# from vector_quantize_pytorch import VectorQuantize


class EmotionClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, num_classes),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.classifier(x)
    


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


class ReferenceEncoder_cls(torch.nn.Module): # Original RefEnc
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
        e_dim: int = 256
    ):
        """Initilize reference encoder module."""
        super(ReferenceEncoder_cls, self).__init__()

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

        self.emotion_classifier = EmotionClassifier(e_dim, 7)

    def forward(self, speech: torch.Tensor, emotions) -> torch.Tensor:
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

        # print("ref_embs", ref_embs.shape)

        emotion_preds = self.emotion_classifier(ref_embs)

        # print('emotion_preds', emotion_preds)

        cls_loss = torch.nn.functional.cross_entropy(emotion_preds, emotions)

        return ref_embs, cls_loss

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
        emotion_preds_m = self.emotion_classifier(z_mel)
        cls_loss_m = torch.nn.functional.cross_entropy(emotion_preds, emotions)
        _, ref_embs = self.gru(hs)  # [num_layers, B, gru_units]

        # print("ref_embs:", ref_embs.size())
        return ref_embs[-1]  # Return last layer's output: [B, gru_units]    


class VectorQuantizer_kmeans(nn.Module):

    def __init__(self, n_e=7, e_dim=128, beta=0.25, usage_threshold=1.0e-9):

        super().__init__()

        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.usage_threshold = usage_threshold

        # 코드북 임베딩
        self.embedding = nn.Embedding(self.n_e, self.e_dim)

        # 일단 랜덤 초기화 (혹은 0 초기화 등)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # usage 버퍼: 각 코드가 얼마나 선택되었는지 추적
        # persistent=False -> state_dict에는 저장되지 않음
        self.register_buffer('usage', torch.ones(self.n_e), persistent=False)

        self.perplexity = None
        self.loss = None
    # ------------------------------------------------------------------

    # 1) 코드북을 k-means로 초기화하는 함수

    # ------------------------------------------------------------------

    def init_codebook_kmeans(self, data, max_iter=100):

        """

        data: Tensor of shape (N, e_dim)

            초기화 시 사용될 샘플들(가능하면 충분히 큰 N)

        """

        data_np = data.detach().cpu().numpy() # sklearn은 numpy array 사용

        # n_clusters = self.n_e 로 맞춰서 실행

        kmeans = KMeans(n_clusters=self.n_e, random_state=0, max_iter=max_iter)

        kmeans.fit(data_np)

        # cluster_centers_ shape = (n_e, e_dim)

        new_centers = kmeans.cluster_centers_



        # 코드북에 반영

        with torch.no_grad():

            self.embedding.weight[:] = torch.from_numpy(new_centers).to(self.embedding.weight.device, dtype=self.embedding.weight.dtype)

        print(f"[init_codebook_kmeans] Updated codebook with k-means centers")


    # ------------------------------------------------------------------

    # 2) dead codes를 k-means로 부분 재초기화

    # ------------------------------------------------------------------
    def reset_dead_codes_kmeans(self, data, max_iter=100):
        """
        data: Tensor shape (M, e_dim)
            이번 epoch (혹은 일정 주기) 동안 모은 latent 샘플
        """
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        num_dead = len(dead_codes)
        if num_dead == 0:
            print("[reset_dead_codes_kmeans] No dead codes. Skip.")
            return

        # ---- 1) n_e개 클러스터를 찾는다 (기존: num_dead 만큼이 아니라 n_e로)
        data_np = data.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_e, random_state=0, max_iter=max_iter)
        kmeans.fit(data_np)
        new_centers = kmeans.cluster_centers_  # shape = (n_e, e_dim)

        # ---- 2) '사용 중'인 코드들(usage >= threshold)을 찾는다
        #       (만약 threshold가 너무 낮으면, usage가 0이 아닌 코드 전부를 사용 중이라고 볼 수도 있음)
        used_codes = torch.nonzero(self.usage >= self.usage_threshold).squeeze(1)
        if len(used_codes) == 0:
            # 만약 '사용 중' 코드가 없다면, 그냥 dead code만큼 임의로 뽑아 할당해도 됨
            print("[reset_dead_codes_kmeans] No used codes found. Using first num_dead centers.")
            selected_idx = list(range(num_dead))
        else:
            # ---- 3) 새로 찾은 각 center가, 기존 '사용 중' codebook들과 얼마나 떨어져 있는지 계산
            #         (가장 가까운 used code와의 최소 거리로 측정)
            centers_t = torch.from_numpy(new_centers).to(
                self.embedding.weight.device, dtype=self.embedding.weight.dtype
            )  # (n_e, e_dim)
            used_w = self.embedding.weight[used_codes]  # (num_used, e_dim)

            # 거리 계산: cdist(centers, used_w), shape = (n_e, num_used)
            dist_matrix = torch.cdist(centers_t, used_w, p=2)
            # 각 center별로 "가장 가까운 used code"와의 거리 (min-dist)
            min_dist_to_used, _ = dist_matrix.min(dim=1)  # shape = (n_e,)

            # ---- 4) min_dist_to_used가 큰 순으로 정렬하여, dead code 개수만큼 선택
            sorted_idx = torch.argsort(min_dist_to_used, descending=True)
            selected_idx = sorted_idx[:num_dead]

        # ---- 5) dead_codes 자리에, 골라진 center들을 할당
        with torch.no_grad():
            self.embedding.weight[dead_codes] = centers_t[selected_idx]

        print(f"[reset_dead_codes_kmeans] Replaced {num_dead} dead codes "
            f"with {num_dead} new centers that are farthest from used codes.")



    def random_restart(self):
        #  randomly restart all dead codes below threshold with random code in codebook
        dead_codes = torch.nonzero(self.usage < self.usage_threshold).squeeze(1)
        rand_codes = torch.randperm(self.n_e)[0:len(dead_codes)]
        with torch.no_grad():
            self.embedding.weight[dead_codes] = self.embedding.weight[rand_codes]


    # ------------------------------------------------------------------

    # 기존 로직들

    # ------------------------------------------------------------------

    def dequantize(self, z):

        """ (indices -> embedding vectors) """

        z_flattened = z.view(-1, self.e_dim)

        z_q = self.embedding(z_flattened).view(z.shape)

        return z_q



    def update_usage(self, min_enc):

        # min_enc: (B, 1) 형태의 index 텐서

        self.usage[min_enc] += 1

        self.usage /= 2.0 # decay



    def reset_usage(self):

        self.usage.zero_()

        print("[reset_usage] usage buffer is now zeroed.")



    def forward(self, z, return_indices=False):

        """

        예시상, 입력 z가 이미 (batch, e_dim)인 것으로 가정(사용자 코드에 맞춰 조정 필요).

        만약 2D/3D 형태라면 shape 변환 과정을 추가해야 함.

        """

        # z shape: (B, e_dim) 가정

        z = z.unsqueeze(1)
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)


        # 거리 계산: (z - e)^2 = z^2 + e^2 - 2 z e

        #   (B, 1) + (n_e,) - 2 * (B, e_dim)*(e_dim, n_e)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight ** 2, dim=1) \
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # argmin	
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1) # shape (B,1)

        # one-hot
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e,
            dtype=z.dtype, device=z.device
        )

        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 양자화된 벡터 z_q
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        z_q = z_q.view(z.shape)

        # usage 업데이트
        self.update_usage(min_encoding_indices)



        # VQ-VAE loss
        #  - codebook 손실: mean(||z_q.detach() - z||^2)
        #  - commitment cost: beta * mean(||z_q - z.detach()||^2)

        self.loss = torch.mean((z_q.detach() - z) ** 2) \
                    + self.beta * torch.mean((z_q - z.detach()) ** 2)

        # Straight-Through Estimator
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)

        self.perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 1, 2).contiguous()
        z_q = z_q.squeeze(-1)

        return z_q, self.loss, min_encoding_indices, self.perplexity


class ResidualVQ_kmeans(torch.nn.Module):
    def __init__(
        self,
        n_e: int = 7,
        e_dim: int = 256,
        num_vq: int = 3,
        beta: float = 0.2,
    ):
        super(ResidualVQ_kmeans, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_vq = num_vq
        self.beta = beta

        # VQ 레이어 반복적으로 생성
        self.vq_layers = torch.nn.ModuleList([
            VectorQuantizer_kmeans(
                n_e=n_e,
                e_dim=e_dim,
            )
            for _ in range(num_vq)
        ])

    def forward(self, input_vector: torch.Tensor, cls_loss) -> torch.Tensor:
        residual = input_vector
        vq_losses = []
        perplexities = []
        quantized_codes = []
        indices_list = []

        # 각 단계의 VQ Layer 처리
        for layer in self.vq_layers:
            quantized, vq_loss, indices, _ = layer(residual)
            
            vq_losses.append(vq_loss)

            # Residual 갱신
            residual = residual - quantized.detach()

            # Perplexity 계산
            with torch.no_grad():
                e_mean = torch.mean(F.one_hot(indices, num_classes=layer.n_e).float(), dim=0)
                perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
                perplexities.append(perplexity)

            # 출력 저장
            quantized_codes.append(quantized)
            indices_list.append(indices)

        # 모든 단계의 손실 합산
        total_vq_loss = sum(vq_losses) + cls_loss

        # 모든 단계의 quantized 코드를 concatenate
        final_quantized = torch.cat(quantized_codes, dim=1)

        codebooks = [quantized_codes[0], quantized_codes[1], quantized_codes[2], quantized_codes[0] + quantized_codes[1] + quantized_codes[2]]



        return final_quantized, total_vq_loss, indices_list, codebooks

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

        self.num_vq = num_vq

        self.vq_layer1 = VectorQuantize(
            dim=e_dim, codebook_size=n_e, codebook_dim=e_dim, kmeans_init=True,
        )
        self.vq_layer2 = VectorQuantize(
            dim=e_dim, codebook_size=n_e, codebook_dim=e_dim, kmeans_init=True,
        )
        self.vq_layer3 = VectorQuantize(
            dim=e_dim, codebook_size=n_e, codebook_dim=e_dim, kmeans_init=True,
        )


    def forward(self, ref_embs: torch.Tensor, cls_loss) -> torch.Tensor:
        residual = ref_embs

        # First VQ layer
        z_q_out_1, indices_1, loss_1 = self.vq_layer1(residual)

        # Update residual
        residual = residual - z_q_out_1

        # Second VQ layer
        z_q_out_2, indices_2, loss_2 = self.vq_layer2(residual)

        # Update residual
        residual = residual - z_q_out_2

        # Third VQ layer
        z_q_out_3, indices_3, loss_3 = self.vq_layer3(residual)

        # Concatenate quantized outputs
        z_q_out = torch.cat([z_q_out_1, z_q_out_2, z_q_out_3], dim=1)  # [B, embedding_dim * 3]

        # Total VQ loss
        vq_loss = loss_1 + loss_2 + loss_3

        vq_loss += cls_loss

        # Collect indices and quantized outputs for each layer
        indices_list = [indices_1, indices_2, indices_3]
        codebooks = [z_q_out_1, z_q_out_2, z_q_out_3, z_q_out_1 + z_q_out_2 + z_q_out_3]

        return z_q_out, vq_loss, indices_1, codebooks


# class ResidualVQ2(torch.nn.Module):
#     def __init__(
#         self,
#         e_dim: int = 128,
#         num_vq: int = 2,
#         beta: float = 0.2,
#     ):
#         super(ResidualVQ2, self).__init__()
#         self.beta = beta
#         self.vq_layer1 = VectorQuantize(
#             dim = 128,
#             codebook_size = 16,
#             decay = 0.8,
#             commitment_weight = 0.25,
#             kmeans_init = True,
#         )
#         self.vq_layer2 = VectorQuantize(
#             dim = 128,
#             codebook_size = 16,
#             decay = 0.8,
#             commitment_weight = 0.25,            
#             kmeans_init = True,
#         )

#     def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
#         residual = input_vector
#         quantized_1, indices_1, commit_loss_1  = self.vq_layer1(residual)

#         vq_loss_1 = torch.mean((residual.detach() - quantized_1)**2) + self.beta * \
#             torch.mean((residual - quantized_1.detach())**2)

#         residual = residual - quantized_1.detach()

#         quantized_2, indices_2, commit_loss_2 = self.vq_layer2(residual)
#         vq_loss_2 = torch.mean((residual.detach() - quantized_2)**2) + self.beta * \
#             torch.mean((residual - quantized_2.detach())**2)
        
#         vq_loss = vq_loss_1 + vq_loss_2

#         quantized = torch.cat([quantized_1, quantized_2], dim=1)

#         # print("input_vector", input_vector.shape)
#         # print("indices_1", indices_1.shape)
#         # print("commit_loss_1", commit_loss_1.shape)

#         all_codes = [quantized_1, quantized_2, quantized]

#         print("indices", indices_1, indices_2)

        
#         return quantized, vq_loss, indices_1, all_codes

class ResidualVQ2(torch.nn.Module):
    def __init__(
        self,
        e_dim: int = 128,
        num_vq: int = 2,
        codebook_size: int = 16,
        beta: float = 0.2,
    ):
        super(ResidualVQ2, self).__init__()
        self.num_vq = num_vq
        self.beta = beta

        # VQ 레이어 반복적으로 생성
        self.vq_layers = torch.nn.ModuleList([
            VectorQuantize(
                dim=e_dim,
                codebook_size=codebook_size,
                decay=0.8,
                commitment_weight=0.25,
                kmeans_init=True,
            )
            for _ in range(num_vq)
        ])

        self.code_to_emotion_map = nn.Parameter(torch.randint(0, 7, (codebook_size,)), requires_grad=False)

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        residual = input_vector
        vq_losses = []
        perplexities = []
        quantized_codes = []
        indices_list = []

        # 각 단계의 VQ Layer 처리
        for layer in self.vq_layers:
            quantized, indices, _ = layer(residual)
            
            # VQ 손실 계산
            vq_loss = torch.mean((residual.detach() - quantized)**2) + \
                      self.beta * torch.mean((residual - quantized.detach())**2)
            vq_losses.append(vq_loss)

            # Residual 갱신
            residual = residual - quantized.detach()

            # Perplexity 계산
            with torch.no_grad():
                e_mean = torch.mean(F.one_hot(indices, num_classes=layer.codebook_size).float(), dim=0)
                perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
                perplexities.append(perplexity)

            # 출력 저장
            quantized_codes.append(quantized)
            indices_list.append(indices)

        # 모든 단계의 손실 합산
        total_vq_loss = sum(vq_losses)

        # 모든 단계의 quantized 코드를 concatenate
        final_quantized = torch.cat(quantized_codes, dim=1)

        logits = self.code_to_emotion_map[indices_list[0]] 

        return final_quantized, total_vq_loss, logits, perplexities



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
        beta: float = 0.2,
    ):
        super(SRVQ3, self).__init__()

        self.ref_encp = ReferenceEncoderDynamic()

        self.ref_encd = ReferenceEncoderDynamic()

        self.ref_ence = ReferenceEncoderDynamic()

        self.RVQp = ResidualVQ2()

        self.RVQd = ResidualVQ2()

        self.RVQe = ResidualVQ2()


    def forward(self, speech: torch.Tensor, p_targets: torch.Tensor, d_targets: torch.Tensor, e_targets: torch.Tensor) -> torch.Tensor:
        z_pitch = self.ref_encp(p_targets.float())
        z_duration = self.ref_encd(d_targets.float())
        z_energy = self.ref_ence(e_targets.float())

        quantized_1, commit_loss_1, indices_1, perplexities_1 = self.RVQp(z_pitch)
        quantized_2, commit_loss_2, indices_2, perplexities_2 = self.RVQd(z_duration)
        quantized_3, commit_loss_3, indices_3, perplexities_3 = self.RVQe(z_energy) 

        quantized = torch.cat([quantized_1, quantized_2, quantized_3], dim=1)        

        commit_loss = commit_loss_1 + commit_loss_2 + commit_loss_3

        codebooks = [quantized_1, quantized_2, quantized_3]

        # print("quantized", quantized)

        return quantized, commit_loss, indices_1, codebooks



class ResidualVQ2_kmeans(torch.nn.Module):
    def __init__(
        self,
        n_e: int = 7,
        e_dim: int = 128,
        num_vq: int = 2,
        beta: float = 0.2,
    ):
        super(ResidualVQ2_kmeans, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.num_vq = num_vq
        self.beta = beta

        # VQ 레이어 반복적으로 생성
        self.vq_layers = torch.nn.ModuleList([
            VectorQuantizer_kmeans(
                n_e=n_e,
                e_dim=e_dim,
            )
            for _ in range(num_vq)
        ])

    def forward(self, input_vector: torch.Tensor) -> torch.Tensor:
        residual = input_vector
        vq_losses = []
        perplexities = []
        quantized_codes = []
        indices_list = []

        # 각 단계의 VQ Layer 처리
        for layer in self.vq_layers:
            quantized, vq_loss, indices, _ = layer(residual)
            
            vq_losses.append(vq_loss)

            # Residual 갱신
            residual = residual - quantized.detach()

            # Perplexity 계산
            with torch.no_grad():
                e_mean = torch.mean(F.one_hot(indices, num_classes=layer.n_e).float(), dim=0)
                perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
                perplexities.append(perplexity)

            # 출력 저장
            quantized_codes.append(quantized)
            indices_list.append(indices)

        # 모든 단계의 손실 합산
        total_vq_loss = sum(vq_losses)

        # 모든 단계의 quantized 코드를 concatenate
        final_quantized = torch.cat(quantized_codes, dim=1)

        return final_quantized, total_vq_loss, indices_list, perplexities

class ReferenceEncoderSRVQ3(torch.nn.Module):
    def __init__(self, e_dim
    ):
        super(ReferenceEncoderSRVQ3, self).__init__()

        self.ref_encm = ReferenceEncoder()
        self.ref_encp = ReferenceEncoder()
        self.ref_ence = ReferenceEncoder()

        print("reference_encoder m,p,e initialized")

        self.emotion_classifier = EmotionClassifier(e_dim//2, 7)
        self.cls_loss = 0

    def forward(self, speech, emotions, p_mel, e_mel):
        # Step 1: Extract pitch information and neutralize it
        z_mel = self.ref_encm(speech.float())

        # Step 2: Extract duration information and neutralize it
        z_pitch = self.ref_encp(p_mel.float())

        # Step 3: Extract energy information and neutralize it
        z_energy = self.ref_ence(e_mel.float())

        # Compute emotion classifier loss (after Reference Encoder)
        emotion_preds_m = self.emotion_classifier(z_mel)
        cls_loss_m = torch.nn.functional.cross_entropy(emotion_preds_m, emotions)

        emotion_preds_p = self.emotion_classifier(z_pitch)
        cls_loss_p = torch.nn.functional.cross_entropy(emotion_preds_p, emotions)

        emotion_preds_e = self.emotion_classifier(z_energy)
        cls_loss_e = torch.nn.functional.cross_entropy(emotion_preds_e, emotions)

        self.cls_loss = (cls_loss_m + cls_loss_p + cls_loss_e) / 2

        return z_mel, z_pitch, z_energy, self.cls_loss

class SRVQ3WithNeutralization(torch.nn.Module):
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
        beta: float = 0.2,
    ):
        super(SRVQ3WithNeutralization, self).__init__()

        self.RVQ1 = ResidualVQ2_kmeans(n_e=n_e)
        self.RVQ2 = ResidualVQ2_kmeans(n_e=n_e)
        self.RVQ3 = ResidualVQ2_kmeans(n_e=n_e)
        


    # def forward(self, speech: torch.Tensor, emotions: torch.Tensor, p_mel: torch.Tensor, e_mel: torch.Tensor) -> torch.Tensor:
    def forward(self, z_mel: torch.Tensor, z_pitch: torch.Tensor, z_energy: torch.Tensor, cls_loss) -> torch.Tensor:
        


        quantized_m, commit_loss_m, indices_m, _ = self.RVQ1(z_mel)
        quantized_p, commit_loss_p, indices_p, _ = self.RVQ2(z_pitch)
        quantized_e, commit_loss_e, indices_e, _ = self.RVQ3(z_energy)
        

        # Combine all quantized representations
        quantized = torch.cat([quantized_m, quantized_p, quantized_e], dim=1)
        
        # Compute total commitment loss
        commit_loss = commit_loss_m + commit_loss_p + commit_loss_e

        # Combine indices for reference
        indices = [indices_m[0], indices_m[1], indices_p[0], indices_p[1], indices_e[0], indices_e[1]]

        # print("commit_loss", commit_loss, "cls_loss", cls_loss)

        vq_loss = commit_loss + cls_loss

        codebooks = [quantized_m, quantized_p, quantized_e, quantized_m + quantized_p + quantized_e]

        # print("indices", indices)

        return quantized, vq_loss, indices, codebooks

if __name__ == "__main__":
    ref_embs = torch.rand((7, 256))
    vq_layer = ResidualVQ(n_e=7, e_dim=256)

    quantized, indices, commit_loss, all_codes = vq_layer(ref_embs, return_all_codes = True)
    print(quantized.size())
