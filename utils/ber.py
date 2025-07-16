import torch
import math

def evaluate_ber(enc, dec, device, K, SNR_dBs, num_batches=100, batch_size=1000):
    enc.eval();  dec.eval()
    ber = {}
    # 총 비트 수 (각 SNR마다 동일)
    total_bits = batch_size * K * num_batches

    with torch.no_grad():
        for snr in SNR_dBs:
            # AWGN σ 계산 (√2 분모 포함)
            EbN0_lin = 10**(snr/10)
            sigma = math.sqrt(1/(2*EbN0_lin))

            errors = 0
            for _ in range(num_batches):
                # 1) 랜덤 비트
                m = torch.randint(0, 2, (batch_size, K)).to(device)
                # 2) 인코딩 → 3) 채널
                c = enc(m.float())
                noise = sigma * torch.randn_like(c)
                c_noisy = c + noise
                # 4) 디코딩
                logits = dec(c_noisy)
                m_hat = logits.argmax(dim=2)
                # 5) 에러 집계
                errors += (m_hat.float() != m.float()).sum().item()

            ber[snr] = errors / total_bits

    return ber