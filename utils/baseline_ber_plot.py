import os
import matplotlib.pyplot as plt
from utils.ber import evaluate_ber

def plot_ber_baseline(models,
                      device,
                      snr_list=None,
                      num_batches=1000,    # 기본 batch 수
                      batch_size=1000,     # 기본 batch_size
                      save_path='figures/ber_compare_both_lambdas.png'):
    """
    models: dict
        key: lambda 값 (float)
        value: tuple(enc16, dec16, enc32, dec32)
    device: torch.device
    snr_list: list of dB 값 (default: range(-5,9))
    num_batches: int, evaluate_ber 안에서 반복할 batch 개수
    batch_size: int, 한 배치당 샘플 수
    save_path: 저장할 파일 경로
    """
    if snr_list is None:
        snr_list = list(range(-5, 9))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(6,4))

    for lam, (enc16, dec16, enc32, dec32) in models.items():
        # λ별 스타일 지정
        if lam == 0.9:
            color = 'blue'
            marker = 's'
        elif lam == 0.0:
            color = 'red'
            marker = 'o'
        else:
            color = 'gray'
            marker = 'o'

        # K=16, K=32 각각 BER 계산 및 그리기
        for K, enc, dec in [(16, enc16, dec16), (32, enc32, dec32)]:
            ber_dict = evaluate_ber(
                enc=enc,
                dec=dec,
                device=device,
                K=K,
                SNR_dBs=snr_list,
                num_batches=num_batches,    # 여기에 넘겨줌
                batch_size=batch_size       # 여기에 넘겨줌
            )
            snrs = sorted(ber_dict.keys())
            vals = [ber_dict[s] for s in snrs]
            linestyle = '--' if K == 16 else '-'
            plt.semilogy(
                snrs, vals,
                marker=marker,
                linestyle=linestyle,
                color=color,
                markerfacecolor='none',     # 속 비운 마커
                markeredgecolor=color,      # 테두리 색
                label=f"K={K}, λ={lam}"
            )

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
