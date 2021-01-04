from .dunet import build_unet
from .Nested_UNet import NestedUNet, UNet


def get_model(config):
    model = None
    if config.model.name == "NestedUNet":
        model = NestedUNet(
            config.model.num_classes,
            config.model.input_channels,
            config.model.deep_supervision)
    if config.model.name == "UNet":
        model = UNet(
            config.model.num_classes, config.model.input_channels
        )
    if config.model.name == "DUNet":
        model = build_unet(unet_type="DU",
                           act_fun="relu",
                           pool_method="conv",
                           normalization="bn",
                           device=config.device,
                           num_gpus=1,
                           ch_in=config.model.input_channels,
                           ch_out=1,
                           filters=(32, 512),
                           print_path=None)
    return model
