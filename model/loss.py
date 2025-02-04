import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def group_consistency_loss(y_true, y_pred):
    """
    y_true: 정답 레이블 시퀀스 (길이 N) - Tensor 또는 numpy array
    y_pred: 예측 레이블 시퀀스 (길이 N) - Tensor 또는 numpy array
    
    return: float (각 라벨 그룹별 일관성 로스의 합)
    """
    # 혹시 y_true, y_pred가 torch.Tensor로 GPU에 올라가 있다면
    # detach + cpu로 옮겨서 numpy로 변환해준다.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    assert len(y_true) == len(y_pred), "두 시퀀스 길이가 다릅니다."
    N = len(y_true)
    if N == 0:
        return 0.0

    total_loss = 0.0
    
    # 타겟에 존재하는 유니크 라벨들을 찾는다.
    unique_labels = np.unique(y_true)
    
    # 라벨별로 해당하는 인덱스를 모은 뒤, 그 인덱스의 예측들이 얼마나 "한 가지 라벨"로 일관적인지 계산
    for label in unique_labels:
        # 해당 라벨을 가지는 위치
        indices = np.where(y_true == label)[0]
        # 그 위치들의 예측값 모음
        seg_pred = y_pred[indices]
        
        # seg_pred 내에서 가장 많이 나온 라벨의 빈도수
        counts = np.bincount(seg_pred)
        max_count = np.max(counts)  # 가장 빈도가 높은 예측 라벨의 횟수
        q_k = max_count / len(seg_pred)  # 해당 라벨 그룹 내에서 "최다 라벨"이 차지하는 비율
        
        # 일관성이 높을수록 q_k는 1에 가까워지고, loss는 0에 가까워짐
        loss_k = 1.0 - q_k
        total_loss += loss_k
    
    return total_loss


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.criterion = nn.CrossEntropyLoss()

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
            min_encoding_indices
        ) = predictions

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
        postnet_mel_predictions = postnet_mel_predictions.masked_select(
            mel_masks.unsqueeze(-1)
        )
        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)

        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        # Style loss 
        style_loss = self.mae_loss(style_pred_embs, style_ref_embs) * 10 # lamda scale
        total_style_loss = style_loss + guided_loss

        classifier_loss = 0
        for min_encoding_indice in min_encoding_indices:
            emotions_pred = min_encoding_indice
            # Clssifier
            # emotions_pred = F.one_hot(emotions_pred, num_classes=7).float().squeeze()

            # emotions_pred = min_encoding_indices.float().squeeze()
            emotions = inputs[3]

            

            # print("emotions", emotions)

            # print("emotions_pred ", emotions_pred)
            # print("emotions ", emotions)

            classifier_loss += group_consistency_loss(emotions_pred, emotions) * 0.1

            # print("-----------------------")

        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + total_style_loss + vq_loss + classifier_loss 
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
        )
