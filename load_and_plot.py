import torch
import os

from utils.baseline_acsl_plot import plot_acsl_baseline
from utils.baseline_ber_plot import plot_ber_baseline
from utils.baseline_loss_plot import plot_loss_baseline
from model.autoencoder.EncDec import Encoder, Decoder

# ──────────── λ 목록 정의 ────────────
lam_list = [0.0, 0.9]
lam_strs = [str(l).replace('.', '') for l in lam_list]

# device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ensure figures dir
os.makedirs('figures', exist_ok=True)

# 모델과 히스토리 저장용 딕셔너리
ae_histories = {}
base_models   = {}

# ──────────── 모든 λ 모델 및 히스토리 로드 ────────────
for lam, lam_str in zip(lam_list, lam_strs):
    # Load training history from .pt
    ae_hist16_file = f"pre-trained/history/ae_history16_lambda{lam_str}.pt"
    ae_hist32_file = f"pre-trained/history/ae_history32_lambda{lam_str}.pt"
    assert os.path.isfile(ae_hist16_file), f"History file not found: {ae_hist16_file}"
    assert os.path.isfile(ae_hist32_file), f"History file not found: {ae_hist32_file}"
    ae_history16 = torch.load(ae_hist16_file, map_location=device)
    ae_history32 = torch.load(ae_hist32_file, map_location=device)
    ae_histories[lam] = (ae_history16, ae_history32)

    # Instantiate AE models (architecture only)
    base_enc16 = Encoder(16, 32).to(device)
    base_dec16 = Decoder(16, 32).to(device)
    base_enc32 = Encoder(32, 64).to(device)
    base_dec32 = Decoder(32, 64).to(device)

    # Load pretrained weights
    pre_dir = 'pre-trained'
    for K, enc, dec in [(16, base_enc16, base_dec16), (32, base_enc32, base_dec32)]:
        enc_path = os.path.join(pre_dir, 'AE', f"ae_encoder{K}_lambda{lam_str}.pth")
        dec_path = os.path.join(pre_dir, 'AE', f"ae_decoder{K}_lambda{lam_str}.pth")
        assert os.path.isfile(enc_path), f"Missing encoder checkpoint: {enc_path}"
        assert os.path.isfile(dec_path), f"Missing decoder checkpoint: {dec_path}"
        enc.load_state_dict(torch.load(enc_path, map_location=device))
        dec.load_state_dict(torch.load(dec_path, map_location=device))

    base_models[lam] = (base_enc16, base_dec16, base_enc32, base_dec32)

# ──────────── Baseline Loss & ACSL: λ별 개별 플롯 ────────────
for lam in lam_list:
    ae_hist16, ae_hist32 = ae_histories[lam]
    # 1) Training loss baseline
    plot_loss_baseline(ae_hist16, ae_hist32, lambda_=lam)
# 2) ACSL baseline (K=16)
plot_acsl_baseline(base_models, device, sample_size=65536)  # 65536 = 2^16
# BER baseline
plot_ber_baseline(base_models, device)
