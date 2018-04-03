###############################################################################
# Torch utils
import torch


def optimzier(method, params, **keys):
    method = method.lower()
    if method == 'adam':
        return torch.optim.Adam(params, **keys)
    elif method == 'adagrad':
        return torch.optim.Adagrad(params, **keys)
    elif method == 'sgd':
        return torch.optim.SGD(params, **keys)
        # return torch.optim.lr_scheduler.StepLR(opt, 5000)
    elif method == 'rmsprop':
        return torch.optim.RMSprop(params, **keys)
    else:
        raise ValueError('Unkonw optimzier method: %s', method)
