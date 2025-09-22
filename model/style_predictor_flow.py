# style_predictor_flow.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Rectified Flow 라이브러리 (clone + editable install 후 import 가능)
from rectified_flow.rectified_flow import RectifiedFlow


# ------------------------- utils -------------------------
def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """x:[B,T,D], mask:[B,T](True=pad) -> [B,D]"""
    if mask is None:
        return x.mean(dim=1)
    lengths = (~mask).sum(dim=1).clamp(min=1).unsqueeze(-1)
    x = x.masked_fill(mask.unsqueeze(-1), 0.0)
    return x.sum(dim=1) / lengths


# style_predictor_flow.py

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden: int = 256, max_freq: int = 16):
        super().__init__()
        self.max_freq = max_freq
        self.proj = nn.Linear(2 * max_freq, hidden)

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: [B,1] in [0,1]
        # [B,1] 보장
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        # freqs를 t와 같은 device/dtype으로
        freqs = torch.arange(
            1, self.max_freq + 1, device=t.device, dtype=t.dtype
        )[None, :] * math.pi  # [1, F]
        angles = t * freqs                        # [B, F]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B, 2F]
        return self.proj(emb)  # [B, hidden]
    

# ----------------- Transformer-based velocity field -----------------
class TransformerVelocityField(nn.Module):
    """
    v_theta(x_t, t, cond): [B, Dx] x [B,1] x (text_ctx:[B,H_t], tag:[B,H_tag], opt spk:[B,H_s]) -> [B, Dx]
    - x_token(=x_t)에 시간 임베딩을 더해 query로 사용
    - cond 토큰(텍스트 요약, 스타일 태그[, 스피커])과 self-attention
    - 최종 x_token hidden을 MLP로 Dx 속도로 투영
    """
    def __init__(
        self,
        dim_x: int,        # style vector dim (Dx)
        d_model: int,      # Transformer hidden
        nhead: int,
        nlayers: int,
        dropout: float,
        use_spk_token: bool = False,
        dim_text_ctx: int = 256,  # 입력: text pooled dim (= encoder_hidden)
        dim_tag: int = 256,       # 입력: style tag dim (emotion emb dim)
        dim_spk: int = 0,         # 입력: speaker emb dim
    ):
        super().__init__()
        self.dim_x = dim_x
        self.use_spk_token = use_spk_token and (dim_spk is not None) and (dim_spk > 0)

        # 토큰 프로젝션
        self.x_proj   = nn.Linear(dim_x, d_model)
        self.txt_proj = nn.Linear(dim_text_ctx, d_model)
        self.tag_proj = nn.Linear(dim_tag, d_model)
        if self.use_spk_token:
            self.spk_proj = nn.Linear(dim_spk, d_model)

        # 시간 임베딩을 x token에 더해줌
        self.t_emb = SinusoidalTimeEmbedding(hidden=d_model)

        # 간단한 위치 임베딩(토큰 수가 작아도 안정성 ↑)
        self.pos_emb = nn.Parameter(torch.zeros(1, 3 + int(self.use_spk_token), d_model))

        # Transformer Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # 출력: d_model -> Dx 속도
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, dim_x),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        text_ctx: torch.Tensor,
        style_tag_emb: torch.Tensor,
        spk_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 모두 x_t와 동일한 device/dtype으로 정렬
        t            = t.to(device=x_t.device, dtype=x_t.dtype)
        text_ctx     = text_ctx.to(device=x_t.device, dtype=x_t.dtype)
        style_tag_emb= style_tag_emb.to(device=x_t.device, dtype=x_t.dtype)
        if self.use_spk_token and (spk_cond is not None):
            spk_cond = spk_cond.to(device=x_t.device, dtype=x_t.dtype)

        x_tok   = self.x_proj(x_t) + self.t_emb(t)
        txt_tok = self.txt_proj(text_ctx)
        tag_tok = self.tag_proj(style_tag_emb)
        tokens = [x_tok, txt_tok, tag_tok]
        if self.use_spk_token and (spk_cond is not None):
            tokens.append(self.spk_proj(spk_cond))

        h = torch.stack(tokens, dim=1)
        h = h + self.pos_emb[:, :h.size(1), :]
        h = self.encoder(h)
        x_hidden = h[:, 0, :]
        return self.out(x_hidden)


# ------------------------ Main module ------------------------
class StylePredictorFlow(nn.Module):
    """
    Rectified Flow 기반 스타일 프리딕터 (Transformer velocity).
    - __init__ 인자는 fastspeech2.py에서 넘기는 것만 사용.
    - forward:
        train  : (x(t_end), flow_loss)  # target_style 주면 RF loss 계산
        infer  : x(t_end)               # x0에서 0->t_end 적분한 결과
    """
    def __init__(
        self,
        dim_text: int = 256,
        dim_tag: int = 256,
        dim_spk: int = 256,
        dim_style: int = 256,
        hidden: int = 256,
        dropout: float = 0.1,
        use_transformer_block: bool = False,  # (호환용; 항상 Transformer 사용)
        nhead: int = 2,
        nlayers: int = 1,
        # ▼ relative noise 계수 k (σ_eff = clamp(k * RMS(spk_emb), 1e-3, 0.5))
        noise_k_train: float = 0.1,
        noise_k_infer: float = 0.0,
    ):
        super().__init__()
        self.dim_style = dim_style
        self.dim_spk   = dim_spk
        assert (self.dim_spk == 0) or (self.dim_spk == self.dim_style), \
            f"dim_spk({self.dim_spk}) must equal dim_style({self.dim_style}) when using spk_emb as x0."

        # 텍스트 요약(고정 길이 cond)
        self.text_norm = nn.LayerNorm(dim_text)

        # spk 토큰을 cond에 추가할지 (필요 시 True)
        self.include_spk_in_cond = False

        # Transformer velocity field
        self.vfield = TransformerVelocityField(
            dim_x=dim_style,
            d_model=hidden,
            nhead=nhead,
            nlayers=nlayers,
            dropout=dropout,
            use_spk_token=self.include_spk_in_cond,
            dim_text_ctx=dim_text,
            dim_tag=dim_tag,
            dim_spk=dim_spk,
        )

        # Rectified Flow 엔진: 직선 보간
        self.rf = RectifiedFlow(
            data_shape=(dim_style,),
            velocity_field=self.vfield,
            interp="straight",                 # x_t=(1-t)x0 + t x1,  xdot = x1 - x0
            is_independent_coupling=True,
        )

        # --- 여기서 criterion 래핑 (패키지 수정 없이 time_weights를 GPU로 강제) ---
        base_criterion = self.rf.criterion
        class _DeviceSafeCriterion:
            def __init__(self, base): self.base = base
            def __call__(self, v_t, dot_x_t, x_t, t, time_weights):
                dev, dt = v_t.device, v_t.dtype
                # time_weights를 v_t 기준으로 캐스팅 + 모양 [B] 보장
                tw = time_weights.to(device=dev, dtype=dt)
                if tw.dim() != 1 or tw.size(0) != v_t.size(0):
                    tw = tw.view(v_t.size(0))
                return self.base(v_t, dot_x_t, x_t, t, tw)
        self.rf.criterion = _DeviceSafeCriterion(base_criterion)

        # relative noise 계수
        self.noise_k_train = float(noise_k_train)
        self.noise_k_infer = float(noise_k_infer)

        # spk_emb가 없는 경우 대비 기본 베이스(학습됨)
        self.fallback_base = nn.Parameter(torch.zeros(1, dim_style))

        # 수치 적분 스텝 (정확도/속도 트레이드오프)
        self.flow_steps = 4

    # ---------- helpers ----------
    def _build_text_ctx(self, text_enc: torch.Tensor, text_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # 텍스트를 global cond로 요약 (masked mean → LN)
        txt = masked_mean(text_enc, text_mask)           # [B, dim_text]
        return self.text_norm(txt)                       # [B, dim_text]

    def _relative_noise(self, spk_emb: torch.Tensor, k: float) -> torch.Tensor:
        """
        σ_eff = clamp(k * RMS(spk_emb), 1e-3, 0.5)
        x0 = spk_emb + σ_eff * N(0, I)
        """
        if k <= 0.0:
            return spk_emb
        # RMS per sample: [B,1]
        rms = spk_emb.detach().pow(2).mean(dim=1, keepdim=True).sqrt()
        sigma_eff = (k * rms).clamp(min=1e-3, max=0.5)   # 보호 클램프
        noise = torch.randn_like(spk_emb) * sigma_eff    # 브로드캐스트로 [B,D]
        return spk_emb + noise

    def _make_x0(self, spk_emb: Optional[torch.Tensor], k: float) -> torch.Tensor:
        """
        spk_emb가 있으면 relative noise로 x0 생성, 없으면 fallback 사용.
        """
        if (spk_emb is not None) and (self.dim_spk > 0):
            x0 = self._relative_noise(spk_emb, k=k)      # ★ 여기서 요청한 방식 적용
        else:
            B = spk_emb.size(0) if spk_emb is not None else 1
            base = self.fallback_base.expand(B, -1)
            # fallback에는 절대 노이즈를 쓰지 않고 그대로 둠(원하면 필요 시 추가)
            x0 = base
        return x0

    def _euler_integrate_to(
        self,
        x0: torch.Tensor,
        text_ctx: torch.Tensor,
        style_tag_emb: torch.Tensor,
        spk_emb: Optional[torch.Tensor],
        t_end: float,
        steps: int,
    ) -> torch.Tensor:
        """
        0 -> t_end 까지 Euler 적분.
        steps는 정확도(수치오차) 제어용. 강도는 t_end로 제어.
        """
        B = x0.size(0)
        device = x0.device
        t_end = float(max(0.0, min(1.0, t_end)))
        steps = max(int(steps), 1)
        h = t_end / steps
        t = 0.0
        x = x0
        for _ in range(steps):
            t_tensor = torch.full((B, 1), t, device=device, dtype=x.dtype)
            v = self.vfield(
                x, t_tensor,
                text_ctx=text_ctx,
                style_tag_emb=style_tag_emb,
                spk_cond=spk_emb if self.vfield.use_spk_token else None,
            )
            x = x + h * v
            t += h
        return x  # x(t_end)

    # ---------- public API ----------
    def forward(
        self,
        text_enc: torch.Tensor,           # [B,T,dim_text]
        style_tag_emb: torch.Tensor,      # [B,dim_tag]
        neu_emb: Optional[torch.Tensor] = None,   # [B,dim_spk] or None
        text_mask: Optional[torch.Tensor] = None, # [B,T] bool
        target_style: Optional[torch.Tensor] = None,  # [B,dim_style] (x1, train에서 주면 RF loss 계산)
        return_loss: bool = False,
        t_end: float = 1.0,               # 0->t_end 적분 (강도 제어)
        steps: Optional[int] = None,      # 적분 스텝(정확도/속도 트레이드오프)
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        train:  target_style!=None and return_loss=True -> (x(t_end), flow_loss)
        infer:  target_style=None  or return_loss=False  -> x(t_end)
        """
        # 1) cond 만들기
        text_ctx = self._build_text_ctx(text_enc, text_mask)  # [B, dim_text]

        # 2) x0 생성 (학습/추론 모드별 relative noise k 선택)
        k = self.noise_k_train if ((target_style is not None) and return_loss) else self.noise_k_infer
        x0 = self._make_x0(spk_emb, k=k).to(text_enc.device)  # [B, dim_style]

        # 3) x(t_end) 계산 (inference 루트)
        integ_steps = steps if (steps is not None) else self.flow_steps
        x_t = self._euler_integrate_to(
            x0=x0,
            text_ctx=text_ctx,
            style_tag_emb=style_tag_emb,
            spk_emb=spk_emb,
            t_end=t_end,
            steps=integ_steps,
        )  # [B, dim_style]

        # 4) train 경로: flow loss 계산 (x0, x1 → 속도장 회귀; t ~ U(0,1))
        if (target_style is not None) and return_loss:
            flow_loss = self.rf.get_loss(
                x_0=x0,
                x_1=target_style,
                text_ctx=text_ctx,
                style_tag_emb=style_tag_emb,
                spk_cond=spk_emb if self.vfield.use_spk_token else None,
            )
            return x_t, flow_loss

        return x_t