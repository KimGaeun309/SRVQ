import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import Optional

def create_triplet_samples(embeddings: torch.Tensor, labels: torch.Tensor):
    """
    embeddings: (B, D)
    labels: (B,)
    반환:
    anchor, positive, negative: 각각 (N, D)  (N <= B, 유효하게 매칭된 쌍 수)
    """
    device = embeddings.device
    B = embeddings.size(0)

    anchor_list   = []
    positive_list = []
    negative_list = []

    for i in range(B):
        anchor_label = labels[i].item()
        same_indices = [j for j in range(B) if j != i and labels[j].item() == anchor_label]
        diff_indices = [j for j in range(B) if labels[j].item() != anchor_label]

        if len(same_indices) > 0 and len(diff_indices) > 0:
            p = random.choice(same_indices)
            n = random.choice(diff_indices)
            anchor_list.append(embeddings[i])
            positive_list.append(embeddings[p])
            negative_list.append(embeddings[n])

    if len(anchor_list) == 0:
        return None, None, None

    anchor   = torch.stack(anchor_list).to(device)
    positive = torch.stack(positive_list).to(device)
    negative = torch.stack(negative_list).to(device)
    return anchor, positive, negative


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"]["feature"]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"]["feature"]

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.criterion = nn.CrossEntropyLoss()
        self.triplet_margin_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

        # ▼ neutral 설정 (없으면 기능 비활성화)
        sp_cfg = model_config.get("style_predictor", {})
        self.neutral_id: Optional[int] = sp_cfg.get("neutral_id", None)
        # neutral ref를 spk_emb에 약간의 RMS 비례 노이즈로 흔들고 싶으면 >0로 설정 (예: 0.05~0.15)
        self.neutral_ref_noise_k: float = float(sp_cfg.get("neutral_ref_noise_k", 0.1))

    @staticmethod
    def _relative_noise(x: torch.Tensor, k: float) -> torch.Tensor:
        """
        σ_eff = clamp(k * RMS(x), 1e-3, 0.5) (per-sample)
        return x + σ_eff * N(0, I)
        """
        if k <= 0.0:
            return x
        rms = x.detach().pow(2).mean(dim=1, keepdim=True).sqrt()     # [B,1]
        sigma_eff = (k * rms).clamp(min=1e-3, max=0.5)               # 안정성 클램프
        noise = torch.randn_like(x) * sigma_eff
        return x + noise

    def forward(self, inputs, predictions, step):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[7:]

        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            style_ref_embs, 
            style_pred_embs,
            guided_loss,
            vq_loss,
            flow_loss,
            min_encoding_indices,
            orig_style_ref_embs,
            neu_emb,                     # ★ 추가: fastspeech2.forward에서 넘겨줌
        ) = predictions

        # ====== 마스크/타겟 정리 ======
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        # ====== 기본 loss들 ======
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # ====== Style loss (neutral 타겟 치환) ======
        # emotions 라벨
        emotions = inputs[3]                     # (B,)
        B = emotions.size(0)
        device = emotions.device

        # style_ref 타겟을 복사한 뒤, neutral이면 spk_emb로 치환(grad 막기 + 선택적 noise)
        style_ref_target = style_ref_embs.detach()
        if (self.neutral_id is not None) and (neu_emb is not None):
            neutral_mask = (emotions == self.neutral_id)
            if neutral_mask.any():
                # detach + optional relative noise
                # spk_target = self._relative_noise(spk_emb.detach(), self.neutral_ref_noise_k)
                neu_target = self._relative_noise(neu_emb.detach(), self.neutral_ref_noise_k)
                # shape 맞추기: style_ref_embs는 [B, D]s
                style_ref_target = style_ref_target.clone()
                style_ref_target[neutral_mask] = neu_target[neutral_mask]

        # predictor MSE (flow 기반 predictor라도 보조 MSE는 regularizer로 유용)
        style_loss = self.mae_loss(style_pred_embs, style_ref_target) * 0.1

        vq_loss = vq_loss * 0.1

        # RF term (모델에서 이미 batch 평균된 scalar로 넘어온다고 가정)
        style_flow_term = flow_loss * 30.0
        guided_loss = guided_loss * 0.1
        total_style_loss = style_loss + guided_loss + style_flow_term

        # ====== Triplet loss (neutral만 stop-grad) ======
        classifier_loss = torch.zeros_like(mel_loss)

        if orig_style_ref_embs is not None:
            # 원 코드의 슬라이스 패턴 유지 (RVQ stage별/합성 등)
            # 주: 이 슬라이스는 프로젝트 설정에 따라 다를 수 있음. 기존 코드를 그대로 따름.
            parts = [
                orig_style_ref_embs[:, :256],
                orig_style_ref_embs[:, 256:512],
                orig_style_ref_embs[:, 256:384],
                orig_style_ref_embs[:, 384:512],
                orig_style_ref_embs[:, 512:],
            ]

            # neutral stop-grad: neutral 행만 detach된 복사본으로 triplet 구성
            if (self.neutral_id is not None):
                neutral_mask = (emotions == self.neutral_id)
            else:
                # neutral id 미설정이면 stop-grad 비활성화
                neutral_mask = torch.zeros(B, dtype=torch.bool, device=device)

            for emb in parts:
                if emb.numel() == 0:
                    continue
                emb_for_triplet = emb
                if neutral_mask.any():
                    emb_for_triplet = emb_for_triplet.clone()
                    # neutral 행만 detach
                    emb_for_triplet[neutral_mask] = emb_for_triplet[neutral_mask].detach()

                anchor, positive, negative = create_triplet_samples(emb_for_triplet, emotions)
                if anchor is not None:
                    classifier_loss = classifier_loss + self.triplet_margin_loss_fn(anchor, positive, negative) * 0.1

        # ====== 총합 ======
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss
            + total_style_loss + vq_loss + classifier_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            style_loss,
            guided_loss,
            vq_loss,
            classifier_loss,
            style_flow_term,
        )
