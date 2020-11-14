from torch.optim import lr_scheduler


def get_scheduler(config, optimizer):
    if config.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs,
            eta_min=config.min_learning_rate)
    elif config.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   factor=config.factor,
                                                   patience=config.patience,
                                                   verbose=1,
                                                   min_lr=config.min_learning_rate)
    elif config.scheduler == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=[
                                                 int(e) for e in config.milestones.split(',')],
                                             gamma=config.gamma)
    elif config.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler
