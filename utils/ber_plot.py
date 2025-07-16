import os
import matplotlib.pyplot as plt
from utils.ber import evaluate_ber

def plot_fig4_compare(b_enc16, b_dec16, b_enc32, b_dec32,
                      i_enc16, i_dec16, i_enc32, i_dec32,
                      device, args,
                      snr_list=None,
                      num_batches=1000, batch_size=1000,
                      save_path=None):
    """
    - base_*: baseline encoder/decoder
    - imp_*: transformer encoder/decoder
    """
    if snr_list is None:
        snr_list = list(range(-5, 9))

    # 1) BER 계산
    bers = {}
    # baseline K=16
    bers['base K=16'] = evaluate_ber(
        enc=b_enc16, dec=b_dec16, device=device,
        K=16, SNR_dBs=snr_list,
        num_batches=num_batches, batch_size=batch_size)
    # improved K=16
    bers['transformer K=16'] = evaluate_ber(
        enc=i_enc16, dec=i_dec16, device=device,
        K=16, SNR_dBs=snr_list,
        num_batches=num_batches, batch_size=batch_size)
    # baseline K=32
    bers['base K=32'] = evaluate_ber(
        enc=b_enc32, dec=b_dec32, device=device,
        K=32, SNR_dBs=snr_list,
        num_batches=num_batches, batch_size=batch_size)
    # improved K=32
    bers['transformer K=32'] = evaluate_ber(
        enc=i_enc32, dec=i_dec32, device=device,
        K=32, SNR_dBs=snr_list,
        num_batches=num_batches, batch_size=batch_size)

    # 2) 저장 경로
    if save_path is None:
        save_path = f'figures/ber_compare.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 3) plot
    plt.figure(figsize=(6,4))
    styles = {
        'base K=16': ('o-',    {'color': 'gray'}),
        'transformer K=16': ('o--',   {'color': 'tab:blue'}),
        'base K=32': ('s-',    {'color': 'lightgray'}),
        'transformer K=32': ('s--',   {'color': 'tab:orange'}),
    }
    for label, ber_dict in bers.items():
        snrs = sorted(ber_dict.keys())
        bers_vals = [ber_dict[s] for s in snrs]
        marker_linestyle, style_kwargs = styles[label]
        plt.semilogy(snrs, bers_vals,
                     marker_linestyle,
                     label=label,
                     **style_kwargs)

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER Curve (λ={args.lam})")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
