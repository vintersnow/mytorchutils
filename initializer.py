def uniform_init(tens, max_value):
    '''
    Args:
        tens (Tensor)
        max_value (int or float)
    '''
    tens.data.uniform_(-max_value, max_value)


def normal_init(tens, mean=0, std=1):
    '''
    Args:
        tens (Tensor)
        mean (int or float)
        std (int or float)
    '''
    tens.data.normal_(mean, std)
