from modules.baseline_model import BaselineNet
from modules.improved_unet_model import ImprovedUNet
from modules.experiment_unet_model import r0ImprovedUNet


def prepare_model(args):
    if args.baseline:
        print("===> prepare BASELINE model ...")
        model = BaselineNet(args)
    else:
        print("===> prepare IMPROVEMENT model ...")
        model = r0ImprovedUNet(args)

    return model
