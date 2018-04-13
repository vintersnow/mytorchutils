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
