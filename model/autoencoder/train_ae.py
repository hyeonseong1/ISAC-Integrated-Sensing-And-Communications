import torch
import torch.optim as optim
import numpy as np

from model.autoencoder.EncDec import Encoder, Decoder
from model.loss_fn import isac_loss
from model.autoencoder.code_init import concatenated_code_initialization

def train(K, N, device, args):
    # torch.manual_seed(0)

    # Hyperparameters (Table I)
    batch_size = 1000
    num_epochs = 400
    Nenc, Ndec = 10, 50    # train for loop 10, 50 times each

    # Naming options
    model = args.model
    lam = args.lam
    lam_str = str(lam).replace('.', '')

    # Call autoencoder
    enc = Encoder(K, N).to(device)
    dec = Decoder(K, N).to(device)
    enc_opt = optim.Adam(enc.parameters(), lr=3e-4)
    dec_opt = optim.Adam(dec.parameters(), lr=3e-4)
    # if K == 16:
    #     enc_opt = optim.Adam(enc.parameters(), lr=3e-4)
    #     dec_opt = optim.Adam(dec.parameters(), lr=3e-4)
    # else:
    #     enc_opt = optim.Adam(enc.parameters(), lr=5e-4)
    #     dec_opt = optim.Adam(dec.parameters(), lr=5e-4)

    # # Scheduler (LR × 0.9 if plateau over 10 epochs, min LR = 1e-6)
    # enc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     enc_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    # )
    # dec_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     dec_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    # )
    enc_sched = torch.optim.lr_scheduler.StepLR(
        enc_opt,
        step_size=10,
        gamma=0.9
    )
    dec_sched = torch.optim.lr_scheduler.StepLR(
        dec_opt,
        step_size=10,
        gamma=0.9
    )

    # Initialization for K=32 using pre-trained 16-bit model
    if K == 32:
        # 1) load 16-bit enc/dec
        tmp_enc = Encoder(K // 2, N // 2).to(device)
        tmp_enc.load_state_dict(torch.load(f"pre-trained/AE/{model}_encoder16_lambda{lam_str}.pth"), strict=True)
        tmp_dec = Decoder(K // 2, N // 2).to(device)
        tmp_dec.load_state_dict(torch.load(f"pre-trained/AE/{model}_decoder16_lambda{lam_str}.pth"), strict=True)

        # 2) apply concatenated code initialization
        concatenated_code_initialization(enc, tmp_enc)  # enc: K=32, tmp_enc: K=16
        concatenated_code_initialization(dec, tmp_dec)

    # loss dict (saved in pre-trained)
    history = {'training': [], 'acsl': [], 'comms': []}

    for epoch in range(num_epochs):
        # SNR scheduling(depend on the paper)
        if K == 16:
            snr = 3 if epoch < 200 else 6
        else:
            snr = 3

        epoch_losses = []
        epoch_acsl = []
        epoch_comms = []

        # --- Encoder update ---
        enc.train(); dec.eval()
        for _ in range(Nenc):
            m = torch.randint(0, 2, (batch_size, K)).float().to(device)
            loss, acsl, comms = isac_loss(enc, dec, m, lam, N, snr)

            epoch_losses.append(loss.item())
            epoch_acsl.append(acsl.item())
            epoch_comms.append(comms.item())

            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()

        # --- Decoder update ---
        enc.eval(); dec.train()
        for _ in range(Ndec):
            m = torch.randint(0, 2, (batch_size, K)).float().to(device)
            loss, acsl, comms = isac_loss(enc, dec, m, lam, N, snr)

            epoch_losses.append(loss.item())
            epoch_acsl.append(acsl.item())
            epoch_comms.append(comms.item())

            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acsl = sum(epoch_acsl) / len(epoch_acsl)
        avg_comms = sum(epoch_comms) / len(epoch_comms)

        # Record losses
        history['training'].append(avg_loss)
        history['acsl'].append(avg_acsl)
        history['comms'].append(avg_comms)

        # enc_sched.step(avg_loss)
        # dec_sched.step(avg_loss)
        enc_sched.step()
        dec_sched.step()

        if epoch % 10 == 0:
            # Convert to dB scale
            tot_db = 10 * np.log10(avg_loss)
            acsl_db = 10 * np.log10(avg_acsl)
            comms_db = 10 * np.log10(avg_comms)
            print(f"Epoch {epoch} | training {tot_db:.2f} dB | acsl {acsl_db:.2f} dB | comms {comms_db:.2f} dB")

    return history, enc, dec