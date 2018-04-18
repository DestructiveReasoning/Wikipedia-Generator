from torch.optim import Adam, RMSprop


def createOptimizer(experiment, params):
    if experiment['type'] == 'Adam':
        return Adam(params)

    if experiment['type'] == 'RMSprop':
        return RMSprop(params, experiment['lr'])
