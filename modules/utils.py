from modules.baseline_model import BaselineNet
from modules.unet_model import UNet


def prepare_model(args):
    if args.baseline:
        print("===> prepare BASELINE model ...")
        model = BaselineNet(args)
    else:
        print("===> prepare IMPROVEMENT model ...")
        model = UNet(args)

    return model
