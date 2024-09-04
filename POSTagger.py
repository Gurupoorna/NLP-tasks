from nltk.tokenize import word_tokenize
import numpy as np
from numba import njit, prange
from viterbi import viterbi_log
from tqdm import tqdm
import logging


class POSTagger():
    logging.basicConfig(level=logging.INFO, format='\n%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    def __init__(self, tagged_sentences=None, words=None, pos_tags=None, this_ST=None, A=None, B=None, Pi=None):
        self._logger = logging.getLogger('POSTagger')
        self._logger.info('Initializing Hidden Markov Model.')

        if not words:
            words = list(set([wt[0] for s in tqdm(tagged_sentences, desc='Getting all words') for wt in s]))
        if not pos_tags:
            pos_tags = list(set([wt[1] for s in tagged_sentences for wt in s]))

        self.words = words
        self.pos_tags = pos_tags
        self._no_words = len(self.words)
        self._no_tags = len(self.pos_tags)
        self._X_id_map = {v: i for i, v in enumerate(self.words)}
        self._Y_id_map = {v: i for i, v in enumerate(self.pos_tags)}

        if isinstance(this_ST, np.ndarray):
            self._ST = this_ST
            self._logger.info('Calculating probability matrices...')
            self._transProb, self._emissProb, self._iniProb = self._get_prob(self.words, self.pos_tags, self._ST)
        elif not A or not B or not Pi:
            self._to_numpy(tagged_sentences)
            self._logger.info('Calculating probability matrices...')
            self._transProb, self._emissProb, self._iniProb = self._get_prob(self.words, self.pos_tags, self._ST)
        else:
            self._transProb = A
            self._emissProb = B
            self._iniProb = Pi

    def tag(self, toks):
        # Tokenize if the input is a string
        if isinstance(toks, str):
            toks = word_tokenize(toks)

        # Generate observation sequence
        O = [self.words.index(tok) if tok in self.words else self._no_words for tok in toks]

        # Use Viterbi algorithm to find optimal sequence of POS tags
        optm_seq, _, _ = viterbi_log(O, self._transProb, self._iniProb, self._emissProb)
        optm_seq = [self.pos_tags[t] for t in optm_seq]

        # Return the tagged sentence as a list of tuples (word, pos_tag)
        test_tagged_sent = list(zip(toks, optm_seq))
        return test_tagged_sent

    def batch_tag(self, X_test):
        if not isinstance(X_test, np.ndarray):
            X_test = self._to_numpy(X_test, give_out=True)
        self._logger.info('Evaluating optimal state sequences using log-viterbi algorithm...')
        return self._viterbi_parll(X_test, self._transProb, self._iniProb, self._emissProb)

    def to_numpy(self, st):
        return self._to_numpy(st, give_out=True)

    def get_ST(self):
        return self._ST

    def _to_numpy(self, XY, give_out=False, to_sort=False):
        if to_sort:
            XY = sorted(XY, key=len)
        n = len(XY)
        m = max(len(s) for s in XY)
        ST = np.zeros((n, m, 2), dtype=np.int32) - 1
        for i, s in enumerate(XY):
            for j, w in enumerate(s):
                ST[i, j] = np.array([self._X_id_map[w[0]], self._Y_id_map[w[1]]])
        if give_out:
            return ST
        self._ST = ST

    @staticmethod
    @njit(parallel=True)
    def _get_prob(words, pos_tags, _ST, eps=0.00001):
        no_words = len(words)
        no_tags = len(pos_tags)
        A = np.zeros((no_tags, no_tags), dtype=np.float64)
        B = np.zeros((no_tags, no_words), dtype=np.float64)
        Pi = np.zeros(no_tags, dtype=np.float64)

        m, n, _ = _ST.shape
        for i in range(m):
            if _ST[i, 0, 0] == -1:
                break
            Pi[_ST[i, 0, 1]] += 1
            for j in range(n - 1):
                wnth = _ST[i, j, 0]
                nth, nnth = _ST[i, j:j + 2, 1]
                B[nth, wnth] += 1
                if nnth == -1:
                    break
                A[nth, nnth] += 1
            if nnth != -1:
                B[_ST[i, n - 1, 1], _ST[i, n - 1, 0]] += 1

        A_s = np.sum(A, axis=1).reshape(-1, 1)
        A /= np.broadcast_to(np.where(A_s == 0, eps, A_s), A.shape)
        B_s = np.sum(B, axis=1).reshape(-1, 1)
        B /= np.broadcast_to(np.where(B_s == 0, eps, B_s), B.shape)
        Pi_s = np.sum(Pi)
        Pi /= np.broadcast_to((eps if Pi_s == 0 else Pi_s), Pi.shape)

        return A, B, Pi

    @staticmethod
    @njit(parallel=True)
    def _viterbi_parll(Xx, A, Pi, B):
        out = np.zeros(Xx.shape, dtype=np.int16) - 1
        for i in prange(Xx.shape[0]):
            O = Xx[i]
            O = O[O > -1]
            out[i, :O.shape[0]], _, _ = viterbi_log(O, A, Pi, B)
        return out