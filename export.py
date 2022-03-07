
from model.LPRNet import build_lprnet
from data.load_data import CHARS
import argparse
import torch


def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', type=int, nargs=2, default=[300, 75], help='the image size')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--cpu', action='store_true', help='Use CPU to load model (default is use cuda)')
    parser.add_argument('--onnx', type=str, default='lprnet.onnx', help='ONNX name')
    parser.add_argument('--weights', default=None, help='pretrained weights')

    args = parser.parse_args()

    return args


def main(args):

    if not args.cpu:
        img = torch.randn((1, 3, args.img_size[1], args.img_size[0]), requires_grad=True).cuda()
    else:
        img = torch.randn((1, 3, args.img_size[1], args.img_size[0]), requires_grad=True)

    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase_train=False, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    print("Model built")

    device = torch.device("cpu" if args.cpu else "cuda:0")
    lprnet.to(device)

    lprnet.load_state_dict(torch.load(args.weights, map_location=torch.device(device)))
    print("Pretrained weights loaded")

    lprnet.eval()
    torch.onnx.export(lprnet,
                      img,
                      args.onnx,
                      export_params=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output']
                      )
    print(f'Onnx model exported at {args.onnx}')


if __name__ == "__main__":
    args = get_parser()
    main(args)