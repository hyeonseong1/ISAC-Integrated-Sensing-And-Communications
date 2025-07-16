import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_baseline(history16, history32, lambda_):
    """
    history*: dict with keys 'total','acsl','comms' containing lists of length num_epochs
    lambda_: float (e.g. 0.9)
    """
    epochs = np.arange(len(history16['training']))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Training (total) loss
    ax.plot(epochs, 10*np.log10(history16['training']), linestyle='-',  label='Training loss (K=16)', color='green')
    ax.plot(epochs, 10*np.log10(history32['training']), linestyle='--', label='Training loss (K=32)', color='green')

    # Communications loss
    ax.plot(epochs, 10*np.log10(history16['comms']), linestyle='-',  label='Comms loss (K=16)', color='purple')
    ax.plot(epochs, 10*np.log10(history32['comms']), linestyle='--', label='Comms loss (K=32)', color='purple')

    # ACSL
    ax.plot(epochs, 10*np.log10(history16['acsl']), linestyle='-',  label='ACSL (K=16)', color='skyblue')
    ax.plot(epochs, 10*np.log10(history32['acsl']), linestyle='--', label='ACSL (K=32)', color='skyblue')

    # SNR step vertical line
    # ax.axvline(200, color='k', linestyle='--', label='SNR step @ epoch 200')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (dB)')
    ax.set_title(f'baseline λ={lambda_}')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    os.makedirs("../figures", exist_ok=True)
    if lambda_ == 0.9:
        plt.savefig('figures/baseline_loss_lambda_09.png')
    else:
        plt.savefig('figures/baseline_loss_lambda_00.png')