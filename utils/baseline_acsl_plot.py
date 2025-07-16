import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator

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
    # stacked: [B, N-1] → lag축 mean → [B]
    return torch.stack(acsls, dim=1).mean(dim=1)


def plot_acsl_baseline(models: dict,
                       device,
                       sample_size: int = 65536,
                       save_path: str = None):
    """
    models: dict
        key: lambda 값 (float)
        value: tuple(enc16, dec16, enc32, dec32)
    device: torch.device
    sample_size: int, ACSL 계산에 사용할 랜덤 메시지 수
    save_path: str or None, 저장 경로
    """
    K = 16
    N = 2 * K
    max_chunk = 1024

    if save_path is None:
        save_path = 'figures/acsl_compare_both_lambdas.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))

    for lam, (enc16, dec16, enc32, dec32) in models.items():
        enc = enc16
        enc.eval()

        # 1) 랜덤 메시지 생성
        B = sample_size
        m = torch.randint(0, 2, size=(B, K), device=device, dtype=torch.float32)

        # 2) 청크 단위 인코딩
        def encode_in_chunks(model, bits):
            chunks = []
            with torch.no_grad():
                for i in range(0, bits.size(0), max_chunk):
                    chunk = bits[i: i + max_chunk]
                    c_chunk = model(chunk)
                    chunks.append(c_chunk.cpu())
                    del c_chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return torch.cat(chunks, dim=0)

        c_all = encode_in_chunks(enc, m)

        # 3) ACSL → dB
        acsl_vals = per_sample_autocorr(c_all, N).numpy()
        eps = 1e-12
        acsl_db = 10 * np.log10(np.clip(acsl_vals, eps, None))
        idx = np.arange(B)
        med = np.median(acsl_db)

        # λ별 스타일
        if lam == 0.9:
            color, marker, linestyle = 'blue', 's', '-'
        else:
            color, marker, linestyle = 'red', 'o', '--'

        # 4) scatter & median line
        ax.scatter(
            idx, acsl_db,
            s=1, alpha=0.4,
            marker=marker,
            facecolors='none',
            edgecolors=color,
            label=f"ML (λ={lam})"
        )
        ax.axhline(
            med, color=color,
            linestyle=linestyle,
            linewidth=1,
            label=f"median: {med:.2f} dB"
        )

    # **여기서 x축을 과학적 표기 + 눈금 개수 설정**
    ax.xaxis.set_major_locator(MaxNLocator(7))                     # 눈금 7개로
    ax.ticklabel_format(style='sci', axis='x', scilimits=(4, 4))   # ×10^4 단위로

    ax.set_title("K=16")
    ax.set_xlabel("Message no.")
    ax.set_ylabel("ACSL (dB)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='upper right')
    fig.tight_layout()

    plt.savefig(save_path)