import torch
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, default='ae',
                    help="ae(autoencoder) vs. tf(transformer)")
parser.add_argument('--lambda', type=float, dest='lam', required=True, default=0.9,
                    help="0.9 vs 0")


args = parser.parse_args()
if args.model == "ae":
    from model.autoencoder.train_ae import train
elif args.model == "tf":
    from model.transformer.train_tf import train


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs("pre-trained", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("pre-trained/history", exist_ok=True)

    lambda_ = args.lam
    lam_str = str(lambda_).replace('.', '')

    # When lambda=0.9
    history16, enc16, dec16 = train(K=16, N=32, device=device, args=args)
    if args.model == "ae":
        os.makedirs("pre-trained/AE", exist_ok=True)
        torch.save(history16, f"pre-trained/history/{args.model}_history16_lambda{lam_str}.pt")
        torch.save(enc16.state_dict(), f"pre-trained/AE/{args.model}_encoder16_lambda{lam_str}.pth")
        torch.save(dec16.state_dict(), f"pre-trained/AE/{args.model}_decoder16_lambda{lam_str}.pth")
    elif args.model == "tf":
        os.makedirs("pre-trained/TF", exist_ok=True)
        torch.save(history16, f"pre-trained/history/{args.model}_history16_lambda{lam_str}.pt")
        torch.save(enc16.state_dict(), f"pre-trained/TF/{args.model}_encoder16_lambda{lam_str}.pth")
        torch.save(dec16.state_dict(), f"pre-trained/TF/{args.model}_decoder16_lambda{lam_str}.pth")


    history32, enc32, dec32 = train(K=32, N=64, device=device, args=args)
    if args.model == "ae":
        torch.save(history32, f"pre-trained/history/{args.model}_history32_lambda{lam_str}.pt")
        torch.save(enc32.state_dict(), f"pre-trained/AE/{args.model}_encoder32_lambda{lam_str}.pth")
        torch.save(dec32.state_dict(), f"pre-trained/AE/{args.model}_decoder32_lambda{lam_str}.pth")
    elif args.model == "tf":
        torch.save(history32, f"pre-trained/history/{args.model}_history32_lambda{lam_str}.pt")
        torch.save(enc32.state_dict(), f"pre-trained/TF/{args.model}_encoder32_lambda{lam_str}.pth")
        torch.save(dec32.state_dict(), f"pre-trained/TF/{args.model}_decoder32_lambda{lam_str}.pth")

