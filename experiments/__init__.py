from torch.optim import Adam


def createOptimizer(experiment, params):
    if experiment['type'] == 'Adam':
        return Adam(params)
