import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train-dataset', required=False)
    parser.add_argument('--train-id-set', required=False)
    parser.add_argument('--use-cuda', required=False, default=True)
    parser.add_argument('--cuda_num', required=False, default='cuda:0')
    args = parser.parse_args()

    if args.use_cuda:
        device = torch.device(args.cuda_num)

def train():
    pass


if __name__ == '__main__':
    main()