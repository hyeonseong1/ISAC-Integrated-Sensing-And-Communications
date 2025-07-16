import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plot_fig5_compare(b_hist, i_hist, K, lambda_, save_path=None):
    """
    Single-K: AE vs TF comparison for one K.
    b_hist, i_hist: dicts with keys 'training','comms','acsl'
    K: int (16 or 32)
    """
    epochs = np.arange(len(b_hist['training']))
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        'training': {'ls': '-', 'base':'AE Training', 'imp':'TF Training'},
        'comms':    {'ls': '--','base':'AE Comms',    'imp':'TF Comms'},
        'acsl':     {'ls': ':', 'base':'AE ACSL',     'imp':'TF ACSL'},
    }
    for metric, opts in styles.items():
        ax.plot(epochs, 10*np.log10(b_hist[metric]), opts['ls'], color='gray', label=f"{opts['base']} (K={K})")
        ax.plot(epochs, 10*np.log10(i_hist[metric]), opts['ls'], color='tab:blue', label=f"{opts['imp']} (K={K})")
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (dB)')
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_title(f'K={K} AE vs TF (λ={lambda_})')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    if save_path is None:
        lam_str = str(lambda_).replace('.', '')
        save_path = f'figures/compare_training_K{K}_lambda{lam_str}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def plot_fig5_compare_all(b_hist16, b_hist32, i_hist16, i_hist32,
                          lambda_, save_dir='figures'):
    """
    All-K: AE vs TF comparison for K=16 and K=32, generating two separate plots.
    """
    # Plot for K=16
    plot_fig5_compare(b_hist16, i_hist16, 16, lambda_,
                      save_path=os.path.join(save_dir, f'training_loss_K16_lambda{str(lambda_).replace('.', '')}.png'))
    # Plot for K=32
    plot_fig5_compare(b_hist32, i_hist32, 32, lambda_,
                      save_path=os.path.join(save_dir, f'training_loss_K32_lambda{str(lambda_).replace('.', '')}.png'))

