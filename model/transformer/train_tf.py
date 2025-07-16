import torch
import torch.optim as optim
import numpy as np

from model.transformer.tf_encdec import TransformerEncoderModel, TransformerDecoderModel
from model.loss_fn import isac_loss
from model.transformer.tf_code_init import transformer_concatenated_initialization


def train(K, N, device, args,
          d_model=128, nhead=4, num_layers=3, dim_ff=512):
    """
    Transformer 기반 ISAC autoencoder 학습 함수 (K,N,λ,device). MLP 버전 train_model과 동일한 인터페이스.
    Returns history dict and (enc, dec) models.
    """
    torch.manual_seed(0)
    # Hyperparameters (Table I)
    batch_size = 1000       # (1000x10 + 1000x50) x 400 = 24M
    num_epochs = 400
    Nenc, Ndec = 10, 50
    model = args.model
    lam = args.lam
    lam_str = str(lam).replace('.', '')

    # 1) Transformer Encoder/Decoder 생성
    enc = TransformerEncoderModel(K=K, N=N,
                                  d_model=d_model,
                                  nhead=nhead,
                                  num_layers=num_layers,
                                  dim_feedforward=dim_ff).to(device)
    dec = TransformerDecoderModel(K=K, N=N,
                                  d_model=d_model,
                                  nhead=nhead,
                                  num_layers=num_layers,
                                  dim_feedforward=dim_ff).to(device)

    enc_opt = optim.Adam(enc.parameters(), lr=1e-4)
    dec_opt = optim.Adam(dec.parameters(), lr=1e-4)
    # LR scheduler
    enc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        enc_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    )
    dec_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dec_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    )

    if K == 32:
    # 1) 16-bit Transformer encoder/decoder 로드
        K16, N16 = 16, N // 2
        tmp_enc = TransformerEncoderModel(K16, N16).to(device)
        tmp_enc.load_state_dict(
            torch.load(f"pre-trained/TF/{model}_encoder16_lambda{lam_str}.pth"),
            strict = True)
        tmp_dec = TransformerDecoderModel(K16, N16).to(device)
        tmp_dec.load_state_dict(
            torch.load(f"pre-trained/TF/{model}_decoder16_lambda{lam_str}.pth"),
            strict = True)
        # 2)  초기화 함수 호출 (enc, dec 둘 다)
        transformer_concatenated_initialization(enc, tmp_enc)
        transformer_concatenated_initialization(dec, tmp_dec)

    # 3) 학습 루프
    history = {'training': [], 'acsl': [], 'comms': []}
    for epoch in range(num_epochs):
        # SNR schedule
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
            loss, acsl, comms = isac_loss(enc, dec, m, lam, snr, N)
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
            loss, acsl, comms = isac_loss(enc, dec, m, lam, snr, N)
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

        enc_sched.step(avg_loss)
        dec_sched.step(avg_loss)

        if epoch % 10 == 0:
            # Convert to dB scale
            tot_db = 10 * np.log10(avg_loss)
            acsl_db = 10 * np.log10(avg_acsl)
            comms_db = 10 * np.log10(avg_comms)
            print(f"Epoch {epoch} | training {tot_db:.2f} dB | acsl {acsl_db:.2f} dB | comms {comms_db:.2f} dB")

    return history, enc, dec
