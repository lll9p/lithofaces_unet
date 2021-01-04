from torch.optim import lr_scheduler


def get_scheduler(config, optimizer):
    if config.schedule.name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.train.epochs,
            eta_min=config.schedule.min_learning_rate)
    elif config.schedule.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=config.schedule.factor,
            patience=config.schedule.patience, verbose=1,
            min_lr=config.schedule.min_learning_rate)
    elif config.schedule.name == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                int(e) for e in config.schedule.milestones.split(',')],
            gamma=config.schedule.gamma)
    elif config.schedule.name == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    return scheduler
