class ToTensor(object):
    '''convert to tensor from list'''

    def __init__(self, key_tens):
        '''
        key_tens (dict): key=str, value=tensor function (e.x. torch.LongTensor)
        '''
        self.key_tens = key_tens

    def __call__(self, sample):
        for key, tens in self.key_tens.items():
            sample[key] = tens(sample[key])
        return sample


class Text2Id(object):
    '''単語をID(int)に変換する.'''

    def __init__(self, vocab, *keys):
        self.vocab = vocab
        self.keys = list(keys)

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = [self.vocab.word2id(w) for w in sample[key]]
        return sample


class Token2Id(object):
    def __init__(self, converter, *keys):
        if isinstance(converter, dict):
            self.converter = lambda x: converter[x]
        elif callable(converter):
            self.converter = converter
        else:
            raise ValueError('converter must be dict or callable')
        self.keys = list(keys)

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = [self.converter(w) for w in sample[key]]
        return sample


class ClipText(object):
    '''textがmx_lenより長かった場合切り取る.'''
    def __init__(self, max_len, *keys):
        self.max_len = max_len
        self.keys = list(keys)

    def clip(self, txt, max_len):
        if len(txt) > max_len:
            return txt[:max_len], max_len
        return txt, len(txt)

    def __call__(self, sample):
        '''txtの長さを保持する`key` + _lenというattrを追加する'''
        for key in self.keys:
            sample[key], sample[key + '_len'] = self.clip(
                sample[key], self.max_len)
        return sample


class Tokenizer(object):
    def __init__(self, tokenizer, *keys):
        assert callable(tokenizer)
        self.tokenizer = tokenizer
        self.keys = list(keys)

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = self.tokenizer(sample[key])
        return sample
