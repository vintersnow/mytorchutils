import sys
import time


class OneLinePrint(object):
    def __init__(self, new_line_marker=' $ '):
        self._mark = new_line_marker
        self._pstr = ''

    def write(self, str):
        if self._pstr == '':
            self._pstr += '\r\033[K'
        else:
            self._pstr += self._mark
        self._pstr += str.replace('\n', self._mark)

        return self

    def flush(self):
        sys.stdout.write(self._pstr)
        sys.stdout.flush()
        self._pstr = ''
        return self

    def __call__(self, first_msg):
        self._pstr = ''
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.flush()


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def start(self):
        self.begin = self.prev = time.time()
        return self

    def lap(self, comment=''):
        now = time.time()
        lap = (now - self.prev) * 1e3
        all = (now - self.begin) * 1e3
        self.prev = now
        if self.verbose:
            print('elapsed time: %f ms. lap: %f ms. %s' % (all, lap, comment))
        return lap, all


class Timeit(object):
    def __init__(self):
        self._stack = []

    def pop(self):
        te = time.time()
        st, tf, em = self._stack.pop()
        lap = (te - tf) * 1e3
        print('%s: %f ms. %s' % (st, lap, em))

    def __call__(self, state, start_msg='', end_msg=''):
        t = time.time()
        if start_msg:
            print(start_msg, end='', flush=True)
        self._stack.append((state, t, end_msg))
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.pop()
