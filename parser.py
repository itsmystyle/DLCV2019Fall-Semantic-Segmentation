from __future__ import absolute_import
import os

import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description="DLCV HW2 in image segmentation using pytorch")

    # Datasets parameters
    parser.add_argument(
        "--data_dir", type=str, default=os.path.join("hw2_data"), help="root path to data directory"
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="number of data loading workers (default: 4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("hw2_data"),
        help="root path to output directory",
    )
    parser.add_argument(
        "--augmentation",
        dest="augmentation",
        action="store_true",
        help="Whether to do data augmentation.",
    )

    # Models parameters
    parser.add_argument("--save_dir", type=str, default="models", help="Where to store the model")
    parser.add_argument("--resume", type=str, default="", help="path to the trained model")

    # Training parameters
    parser.add_argument("--epochs", default=100, type=int, help="num of validation iterations")
    parser.add_argument("--train_batch", default=8, type=int, help="train batch size")
    parser.add_argument("--test_batch", default=8, type=int, help="test batch size")
    parser.add_argument("--lr", default=2e-4, type=float, help="initial learning rate")
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, help="initial weight decay rate"
    )
    parser.add_argument(
        "--accumulate_gradient",
        type=int,
        default=1,
        help="Accumulate how many steps of gradient before backpropagate",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="Whether to use pretrained weight",
    )
    parser.add_argument(
        "--baseline", dest="baseline", action="store_true", help="Whether to use baseline model"
    )

    # others
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    return args
