import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def per_sample_autocorr(c: torch.Tensor, N: int) -> torch.Tensor:
    """
    배치 c: [B, 2*N] 의 각 샘플별 ACSL 값을 계산해 반환합니다 (shape [B]).
    """
    re, im = c[:, :N], c[:, N:]
    c_complex = re + 1j * im                # [B, N]
    c_norm = c_complex / torch.linalg.norm(
        c_complex, axis=1, keepdims=True    # [B, N]
    )
    acsls = []
    for l in range(1, N):
        shifted = torch.roll(c_norm, shifts=l, dims=1)  # [B, N]
        corr    = (c_norm.conj() * shifted).sum(dim=1)  # [B]
        acsls.append(torch.abs(corr) ** 2)              # [B]
    # stacked: [B, N-1] → mean over lag축 → [B]
    return torch.stack(acsls, dim=1).mean(dim=1)


def plot_fig2_compare(b_enc16: torch.nn.Module,
                      i_enc16: torch.nn.Module,
                      device,
                      args,
                      full_enum: bool = False,
                      sample_size: int = 65536,
                      save_path: str = None):
    K = 16
    N = 2 * K
    b_enc16.eval()
    i_enc16.eval()

    # 1) 랜덤 메시지
    B = sample_size
    m = torch.randint(0, 2, size=(B, K), device=device, dtype=torch.float32)

    # 2) 인코딩 (청크 단위 처리 + 즉시 CPU 오프로딩)
    max_chunk = 1024
    def encode_in_chunks(enc, bits):
        enc.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, bits.size(0), max_chunk):
                chunk   = bits[i: i + max_chunk]     # [<=max_chunk, K]
                c_chunk = enc(chunk)                 # [<=max_chunk, 2*N] on GPU
                chunks.append(c_chunk.cpu())         # 즉시 CPU로 이동
                del c_chunk
                torch.cuda.empty_cache()
        return torch.cat(chunks, dim=0)            # [B, 2*N] on CPU

    # 3) 코드워드 → 샘플별 ACSL → dB 변환
    c_base = encode_in_chunks(b_enc16, m)         # [B, 2*N]
    c_imp  = encode_in_chunks(i_enc16, m)         # [B, 2*N]

    # per-sample ACSL: both are torch.Tensor [B]
    acsl_base = per_sample_autocorr(c_base, N).numpy()
    acsl_imp  = per_sample_autocorr(c_imp,  N).numpy()

    # log 변환, 0 이하 값(혹은 음수 분산)으로 인한 NaN 방지
    eps = 1e-12
    acsl_db_base = 10 * np.log10(np.clip(acsl_base, eps, None))
    acsl_db_imp  = 10 * np.log10(np.clip(acsl_imp,  eps, None))

    idx = np.arange(B)

    # 4) 플롯
    plt.figure(figsize=(8, 4))
    plt.scatter(idx, acsl_db_base, s=1, alpha=0.4, label='Baseline', color='gray')
    plt.scatter(idx, acsl_db_imp,  s=1, alpha=0.4, label='Improved', color='tab:orange')

    med_base = np.median(acsl_db_base)
    med_imp  = np.median(acsl_db_imp)
    plt.axhline(med_base, linestyle='--', label=f"base median = {med_base:.2f} dB")
    plt.axhline(med_imp,  linestyle='-',  label=f"imp  median = {med_imp:.2f} dB")

    plt.title(f"ACSL Distribution (λ={args.lam})")
    plt.xlabel("Message index")
    plt.ylabel("ACSL (dB)")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    # 5) 저장
    if save_path is None:
        save_path = f'figures/acsl_compare_K16.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
