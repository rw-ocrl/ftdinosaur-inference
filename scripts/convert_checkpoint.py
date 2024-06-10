"""Script converting a checkpoint trained with OCLF to a checkpoint loadable by this codebase."""

import argparse

import torch

from ftdinosaur_inference import utils


def main():
    parser = argparse.ArgumentParser(description="Convert OCLF checkpoint.")
    parser.add_argument(
        "--load-path",
        type=str,
        required=True,
        help="Path to the input checkpoint file.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the converted checkpoint file.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.load_path)
    state_dict = utils.convert_checkpoint_from_oclf(checkpoint)
    torch.save(state_dict, args.save_path)
    print(f"Checkpoint saved to {args.save_path}")


if __name__ == "__main__":
    main()
