from torch import optim


def get_optimizer(config, params):
    if config.optim.name == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.optim.learning_rate,
            weight_decay=config.optim.weight_decay)
    elif config.optim.name == 'SGD':
        optimizer = optim.SGD(params, lr=config.optim.learning_rate,
                              momentum=config.optim.momentum,
                              nesterov=config.optim.nesterov,
                              weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
