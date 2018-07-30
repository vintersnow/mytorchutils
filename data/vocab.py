from os import path, sys
import numpy as np

ap = path.dirname(path.abspath(__file__))  # dataloader
root = path.dirname(ap)  # root
sys.path.append(root)

from logger import get_logger

logger = get_logger(__name__)

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '_PAD_'
UNKNOWN_TOKEN = '_UNK_'
START_DECODING = '_START_'
STOP_DECODING = '_STOP_'


class Vocab(object):
    """単語とidをmappingするクラス"""

    def __init__(self, vocab_file, max_size, min_feq=0, lower=False):
        """
        Args:
            Vocab_file: 語彙ファイルまでのpath. 語彙ファイルの各行は"<word>
            <frequency>"という形式でfrequency順のソートされている前提.
            max_size: 使用する最大単語する（頻度順）. 0の時は全単語を使用.
            lower (bool): Trueなら、全て小文字化する.
        """

        self._word_to_id = {}
        self._id_to_word = {-1: ''}
        self._counter = 0

        self.special_tokens = special_tokens = [
            PAD_TOKEN, START_DECODING, STOP_DECODING
        ]
        for w in special_tokens:
            self._add_word(w)
        self.special_ids = [self.word2id(w) for w in self.special_tokens]

        self._add_word(UNKNOWN_TOKEN)

        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                pieces = line.split()
                if len(pieces) != 1 and len(pieces) != 2:
                    logger.warning(
                        'Incorrect formatted line in vocabulary file: {}\n'.
                        format(line))
                    continue
                w = pieces[0]
                if lower:
                    w = w.lower()
                if w in special_tokens:
                    raise Exception(
                        'A word "{}" conflicts with special_tokens'.format(w))
                if w in self._word_to_id:
                    continue
                    # raise Exception('Dupilcated word: {}'.format(w))
                if len(pieces) >= 2 and int(pieces[1]) <= min_feq:
                    # skip rare word
                    continue

                self._add_word(w)

                if max_size != 0 and self._counter >= max_size:
                    logger.info(
                        'Reached to max size of vocabulary: {}. Stop reading'.
                        format(max_size))
                    break
        logger.debug('Finished loading vocabulary')

        self.pad_id = self.word2id(PAD_TOKEN)
        self.unk_id = self.word2id(UNKNOWN_TOKEN)
        self.start_id = self.word2id(START_DECODING)
        self.stop_id = self.word2id(STOP_DECODING)

    def _add_word(self, word):
        self._word_to_id[word] = self._counter
        self._id_to_word[self._counter] = word
        self._counter += 1

    def word2id(self, word):
        """word(string)に対応するid(integer)を返す"""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, id):
        """id(integer)に対応するword(string)を返す"""
        if id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % id)
        return self._id_to_word[id]

    @property
    def size(self):
        return self._counter


def restore_text(data, vocab, skips=[]):
    """
    Args:
        data: (sequence_length) list or ndarray include word id (integer)
        vocab:
    """
    assert type(data).__module__ == np.__name__
    stop_idx = np.where(data == vocab.stop_id)[0]
    if len(stop_idx) > 0:
        data = data[:stop_idx[0]]
    text = ' '.join([vocab.id2word(id) for id in data if int(id) not in skips])
    return text
