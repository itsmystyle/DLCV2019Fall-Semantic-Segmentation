from baseline_model import BaselineNet
from unet_model import UNet


def prepare_model(args):
    if args.baseline:
        print("===> prepare baseline model ...")
        model = BaselineNet(args)
    else:
        print("===> prepare improvement model ...")
        model = UNet(args)

    return model
