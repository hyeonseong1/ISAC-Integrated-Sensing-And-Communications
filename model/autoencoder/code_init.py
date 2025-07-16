# import torch
#
# def concatenated_code_initialization(model32: torch.nn.Module, model16: torch.nn.Module):
#     """
#     copy model16(dec16) parameters on model32(dec16)
#     W_32 =  [W_16    0]  |  b_32 = [b_16, b_16]
#             [0    W_16]  |
#     """
#     with torch.no_grad():
#         for (_, p16), (_, p32) in zip(model16.named_parameters(), model32.named_parameters()):
#             if p16.ndim == 2:
#                 p32.copy_(torch.block_diag(p16, p16))
#             else:
#                 p32.copy_(torch.cat([p16, p16], dim=0))
#
#         for (n16, buf16), (n32, buf32) in zip(model16.named_buffers(), model32.named_buffers()):
#             if "running_mean" in n16 or "running_var" in n16:
#                 buf32.copy_(torch.cat([buf16, buf16], dim=0))
#             else:
#                 buf32.zero_()
#
#         for (n16, p16), (n32, p32) in zip(
#                 ((n, p) for n, p in model16.named_parameters() if "bn" in n),
#                 ((n, p) for n, p in model32.named_parameters() if "bn" in n),
#         ):
#             p32.copy_(torch.cat([p16, p16], dim=0))

import torch
import torch.nn as nn


def concatenated_code_initialization(big_model: nn.Module, small_model: nn.Module):
    """
    big_model:  K_big = 2 * K_small, hidden dims also 2x 작은 모델
    small_model: pretrained on K_small

    이 함수는 두 모델의 nn.Sequential(net) 안에 있는 nn.Linear 계층을 찾아서,
    small_model 의 weight/bias 를 block‐diagonal 로 복사합니다.
    """
    for (name_b, module_b), (name_s, module_s) in zip(
            big_model.named_modules(), small_model.named_modules()):
        # 둘 다 Linear layer 일 때만 처리
        if isinstance(module_b, nn.Linear) and isinstance(module_s, nn.Linear):
            W_small, b_small = module_s.weight.data, module_s.bias.data
            out_s, in_s = W_small.shape
            W_big, b_big = module_b.weight.data, module_b.bias.data
            out_b, in_b = W_big.shape

            # case A) both out, in dims double: block‐diagonal
            if out_b == 2 * out_s and in_b == 2 * in_s:
                zero = torch.zeros_like(W_small)
                top = torch.cat([W_small, zero], dim=1)  # [out_s, 2*in_s]
                bottom = torch.cat([zero, W_small], dim=1)  # [out_s, 2*in_s]
                W_init = torch.cat([top, bottom], dim=0)  # [2*out_s, 2*in_s]

                b_init = torch.cat([b_small, b_small], dim=0)  # [2*out_s]

                module_b.weight.data.copy_(W_init)
                module_b.bias.data.copy_(b_init)

            # case B) out double, in same: 그냥 위아래로 쌓기
            elif out_b == 2 * out_s and in_b == in_s:
                W_init = torch.cat([W_small, W_small], dim=0)  # [2*out_s, in_s]
                b_init = torch.cat([b_small, b_small], dim=0)  # [2*out_s]

                module_b.weight.data.copy_(W_init)
                module_b.bias.data.copy_(b_init)

            # case C) dims이 같으면 그냥 통으로 복사
            elif out_b == out_s and in_b == in_s:
                module_b.weight.data.copy_(W_small)
                module_b.bias.data.copy_(b_small)

            # 그 외 구조 차이는 무시

