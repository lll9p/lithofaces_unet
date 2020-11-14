from torch import optim


def get_optimizer(config, params):
    if config.optimizer == 'Adam':
        optimizer = optim.Adam(
            params, lr=config.learning_rate,
            weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.learning_rate,
                              momentum=config.momentum,
                              nesterov=config.nesterov,
                              weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer
