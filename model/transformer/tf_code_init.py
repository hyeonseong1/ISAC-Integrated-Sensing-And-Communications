import torch


def transformer_concatenated_initialization(model32: torch.nn.Module, model16: torch.nn.Module):
    """
    Transformer concatenated code initialization.
    - model16: TransformerEncoderModel/TransformerDecoderModel with K=16
    - model32: TransformerEncoderModel/TransformerDecoderModel with K=32 <- initialize this

    Depend on Definition 2:
        copy enc16(dec16) parameters on enc32(dec16)
        W_32 =  [W_16    0]  |  b_32 = [b_16, b_16]
                [0    W_16]  |
    """
    with torch.no_grad():
        for (name16, p16), (name32, p32) in zip(model16.named_parameters(), model32.named_parameters()):
            if p32.shape == p16.shape:
                p32.copy_(p16)
            elif p16.ndim == 2 and p32.shape == (2 * p16.shape[0], 2 * p16.shape[1]):
                blk = torch.block_diag(p16, p16)
                p32.copy_(blk)
            elif p16.ndim == 1 and p32.shape == (2 * p16.shape[0],):
                cat = torch.cat([p16, p16], dim=0)
                p32.copy_(cat)
            else:
                continue

        # copy BN layer
        for (n16, buf16), (n32, buf32) in zip(model16.named_buffers(), model32.named_buffers()):
            if "running_mean" in n16 or "running_var" in n16:
                buf32.copy_(torch.cat([buf16, buf16], dim=0))
            else:
                buf32.zero_()

        # ★ 추가: BatchNorm affine 파라미터 복제
        for (n16, p16), (n32, p32) in zip(
                ((n, p) for n, p in model16.named_parameters() if "bn" in n),
                ((n, p) for n, p in model32.named_parameters() if "bn" in n),
        ):
            p32.copy_(torch.cat([p16, p16], dim=0))