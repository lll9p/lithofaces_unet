from .dunet import build_unet
from .Nested_UNet import NestedUNet, UNet


def get_model(config):
    model = None
    if config.model == "NestedUNet":
        model = NestedUNet(
            config.num_classes, config.input_channels, config.deep_supervision
        )
    if config.model == "UNet":
        model = UNet(
            config.num_classes, config.input_channels
        )
    if config.model == "DUNet":
        model = build_unet(unet_type="DU",
                           act_fun="relu",
                           pool_method="conv",
                           normalization="bn",
                           device="cuda",
                           num_gpus=1,
                           ch_in=config.input_channels,
                           ch_out=1,
                           filters=(32, 512),
                           print_path=None)
    return model
