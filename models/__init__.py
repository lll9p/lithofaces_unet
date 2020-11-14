from .Nested_UNet import NestedUNet, UNet


def get_model(config):
    model = None
    if config.model == "NestedUNet":
        model = NestedUNet(config.num_classes,
                           config.input_channels,
                           config.deep_supervision)
    return model
