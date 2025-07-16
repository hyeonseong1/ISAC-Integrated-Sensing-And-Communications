import math
import torch

def isac_loss(encoder, decoder, m, λ, N, snr_db):
    B, K = m.shape

    # 1) 인코딩
    c = encoder(m)  # [B, 2N]
    # c = c / c.norm(dim=1, keepdim=True)  # ← unit‐power

    # 2) ACSL term --------------------------------------------------------
    re, im = c[:, :N], c[:, N:]
    c_cpx = re + 1j * im                                 # [B, N]

    acsl_vals = []
    for l in range(1, N):
        shifted = torch.roll(c_cpx, shifts=l, dims=1)    # [B, N]
        corr = (c_cpx * shifted.conj()).sum(dim=1)       # [B]
        denom = (c_cpx * c_cpx.conj()).sum(dim=1)        # [B]
        normalized_corr = corr / denom                   # [B]
        acsl_vals.append(torch.abs(normalized_corr)**2)
    acsl_per_sample = torch.stack(acsl_vals, dim=1).mean(dim=1)
    acsl = acsl_per_sample.mean()                        # [B]

    # 3) AWGN noise injection --------------------------------------------
    EbN0_lin = 10 ** (snr_db / 10)
    R = K / N
    sigma = math.sqrt(1 / (2 * EbN0_lin * R))
    noise = sigma * torch.randn_like(c)  # real/imag 모두 동일 분산
    c_noisy = c + noise

    # comms loss: CrossEntropyLoss expects target long labels
    probs = decoder(c_noisy)        # [B, K, 2]
    comm = (-m * torch.log2(probs[:, :, 1]) -(1 - m) * torch.log2(probs[:, :, 0])).mean()

    # 5) 최종 결합
    loss = λ * acsl + (1.0 - λ) * comm
    return loss, acsl.detach(), comm.detach()
