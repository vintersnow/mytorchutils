from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import torch


def padding(s, max_len, value=0):
    '''
    s (Tensor): paddingする元のデータ. 長さはmax_len以下であること.
    max_len (int): padding後の長さ
    value (type(s)): paddingする際に使う値. sのdata tyepと一致する型.
    '''
    pad_size = max_len - s.size(0)
    padded = F.pad(s, (0, pad_size), value=value).data
    return padded, pad_size


def mask(pad_size, max_len, value=1):
    '''Args:
    pad_size (int): paddingの長さ
    max_len (int): 全体の長さ
    value (0 or 1): padding部分の値. default=1
    '''
    return torch.ByteTensor([0] * (max_len - pad_size) + [1] * pad_size)


def merge_samples(samples,
                  text_key=[],
                  ignore_key=[],
                  sortby=None,
                  pad_value=0,
                  make_mask=False):
    '''
    samples (list[sample])
    text_key (list[str]): padding and mask
    ignore_key (list[str]): skip merging to a tensor
    sortby (lambda or None): sort samples in mini-batch
    '''

    if sortby:
        samples = sorted(samples, key=sortby)

    for key in text_key:
        max_len = max(samples, key=lambda x: x[key].size(0))[key].size(0)
        for sample in samples:
            sample[key], pad = padding(sample[key], max_len, pad_value)
            if make_mask:
                sample[key + '_mask'] = mask(pad, max_len)

    batch = {}
    for key in samples[0]:
        if key in ignore_key:
            batch[key] = [s[key] for s in samples]
        else:
            batch[key] = default_collate([s[key] for s in samples])

    return batch


def merge_fn(text_key=[],
             ignore_key=[],
             sortby=None,
             pad_value=0,
             make_mask=False):
    '''make a collage_fn for text dataset
    text_key (list[str]): padding and mask
    ignore_key (list[str]): skip merging to a tensor
    sortby (lambda or None): sort samples in mini-batch
    padding (function(Tensor, max_len(int))): pad the Tensor
    make_mask (function(pad_size(int), max_len(int))): making mask
    '''

    def merge(samples):
        return merge_samples(samples, text_key, ignore_key, sortby, pad_value,
                             make_mask)

    return merge
