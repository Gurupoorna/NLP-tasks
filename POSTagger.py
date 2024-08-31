from nltk.tokenize import word_tokenize

import numpy as np
from numba import jit, njit, prange

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from viterbi import viterbi_log

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO,
        format='\n%(asctime)s [%(levelname)s] %(name)s - %(message)s')

class POSTagger():
    def __init__(self, tagged_sentences:list[list[tuple[str,str]]]=None, 
                 words=None, pos_tags=None, this_ST=None, A=None, B=None, Pi=None):
        self._logger = logging.getLogger('POSTagger')
        self._logger.info('Initializing Hidden Markov Model.')
        if not words:
            words = list(set([wt[0] for s in tqdm(tagged_sentences, desc='Getting all words ') for wt in s]))
        if not pos_tags: pos_tags = list(set([wt[1] for s in tagged_sentences for wt in s]))
        self.words = words
        self.pos_tags = pos_tags
        self._no_words = len(self.words)
        self._no_tags = len(self.pos_tags)
        self._X_id_map = {v: i for i, v in enumerate(self.words)}
        self._Y_id_map = {v: i for i, v in enumerate(self.pos_tags)}
        if isinstance(this_ST,np.ndarray):
            self._ST = this_ST
            self._logger.info('Calculating probability matrices...')
            self._transProb, self._emissProb, self._iniProb = self._get_prob(self.words, self.pos_tags, self._ST)
        elif not A or not B or not Pi:
            self._to_numpy(tagged_sentences)
            self._logger.info('Calculating probability matrices...')
            self._transProb, self._emissProb, self._iniProb = self._get_prob(self.words, self.pos_tags, self._ST)
        else :
            self._transProb = A
            self._emissProb = B
            self._iniProb = Pi
        
    def tag(self, toks):
        if isinstance(toks, str) :
            toks = word_tokenize(toks)
        O = [self.words.index(tok) if tok in self.words else self._no_words for tok in toks]
        optm_seq, _, _ = viterbi_log(O, self._transProb, self._iniProb, self._emissProb)
        optm_seq = [self.pos_tags[t] for t in optm_seq]
        test_tagged_sent  = list(zip(toks,optm_seq))
        return test_tagged_sent

    def batch_tag(self, X_test):
        if not isinstance(X_test,np.ndarray):
            X_test = self._to_numpy(X_test, give_out=True)
        self._logger.info('Evaluating optimal state sequences using log-viterbi algorithm...')
        return self._viterbi_parll(X_test, self._transProb, self._iniProb, self._emissProb)

    def to_numpy(self, st):
        return self._to_numpy(st, give_out=True)
    
    def get_ST(self):
        return self._ST
    
    def _to_numpy(self,XY:list[list[tuple[str,str]]], give_out=False, to_sort=False):
        if to_sort: XY = sorted(XY,key=len)
        n = len(XY)
        m = max(len(s) for s in XY)
        ST = np.zeros((n, m, 2), dtype=np.int32)-1
        for i, s in enumerate(XY):
            for j, w in enumerate(s):
                ST[i, j] = np.array([self._X_id_map[w[0]], self._Y_id_map[w[1]]])
        if give_out: return ST
        self._ST = ST

    @staticmethod
    @njit(parallel=True)
    def _get_prob(words, pos_tags, _ST, eps:float=0.00001):
        no_words = len(words)
        no_tags = len(pos_tags)
        A = np.zeros((no_tags,no_tags), dtype=np.float64)
        B = np.zeros((no_tags,no_words), dtype=np.float64)
        Pi = np.zeros(no_tags, dtype=np.float64)
    
        m, n, _ = _ST.shape
        for i in range(m):
            if _ST[i,0,0] == -1 : break
            Pi[_ST[i,0,1]] += 1
            for j in range(n-1):
                wnth = _ST[i,j,0]
                nth, nnth = _ST[i,j:j+2,1]
                B[nth, wnth] += 1
                if nnth == -1 : break
                A[nth,nnth] += 1
            if nnth != -1 : B[_ST[i,n-1,1], _ST[i,n-1,0]] += 1
                           
        A_s  = np.sum(A,axis=1).reshape(-1,1)  ; A  /= np.broadcast_to(np.where(A_s==0, eps, A_s),A.shape)
        B_s  = np.sum(B,axis=1).reshape(-1,1)  ; B  /= np.broadcast_to(np.where(B_s==0, eps, B_s),B.shape)
        Pi_s = np.sum(Pi)                      ; Pi /= np.broadcast_to((eps if Pi_s==0 else Pi_s),Pi.shape)

        return A, B, Pi
    
    @staticmethod
    @njit(parallel=True)
    def _viterbi_parll(Xx, A, Pi, B):
        out = np.zeros(Xx.shape, dtype=np.int16)-1
        for i in prange(Xx.shape[0]):
            O = Xx[i]
            O = O[O>-1]
            out[i,:O.shape[0]], _, _ = viterbi_log(O, A, Pi, B)
        return out


# @jit(forceobj=True, parallel=True)
def do_kfold(kf_idx , k, words, pos_tags, ST:np.ndarray):
# def do_kfold(k_train_index:np.ndarray, k_test_index:np.ndarray, ST:np.ndarray):
    # Split data into training and testing sets
    no_pos_tags = len(pos_tags)
    precisions = np.empty((k, no_pos_tags), dtype=np.float64)
    recalls    = np.empty((k, no_pos_tags), dtype=np.float64)
    fscores    = np.empty((k, no_pos_tags), dtype=np.float64)
    c_matrices = np.empty((k, no_pos_tags, no_pos_tags), dtype=np.int_)
    i = 0
    for train_index, test_index in kf_idx:
        train_ST = ST[train_index]
        test_ST =  ST[test_index]
        test_ST_o, test_ST_q = np.moveaxis(test_ST,-1,0)
        # Making model
        HMM_POSTagger_k = POSTagger(words=words, pos_tags=pos_tags, this_ST=train_ST)
        # Testing
        pred = HMM_POSTagger_k.batch_tag(test_ST_o)
        assert ((pred == -1) == (test_ST_q == -1)).all()
        y_true = test_ST_q[test_ST_q!=-1].ravel()
        y_pred = pred[pred!=-1].ravel()
        conf_matx = confusion_matrix(y_true, y_pred)
        precision_s, recall_s, fscore_s, _ = precision_recall_fscore_support(
            y_true, y_pred, 
            beta=1, 
            labels=range(no_pos_tags)
        )
        # Per label accuracy which i think turns out same as precision
        # prec_s = conf_matx.sum(axis=0)
        # prec_s[prec_s==0] = 1 # handling division by zero
        # precis = conf_matx.diagonal() / prec_s

        # # Per label recall
        # rec_s = conf_matx.sum(axis=1)
        # rec_s[rec_s==0] = 1 # handling division by zero
        # recl = conf_matx.diagonal() / rec_s

        # print('precision matchs' if (precis == precision_s).all() else 'precision UNmatched')
        # print('recall matchs' if (recl == recall_s).all() else 'recall UNmatched')

        precisions[i] = precision_s
        recalls[i]    = recall_s
        fscores[i]    = fscore_s
        c_matrices[i] = conf_matx
        i+=1
    
    return precisions, recalls, fscores, c_matrices

if __name__ == '__main__' :
    import sys
    from nltk.corpus import brown
    import pandas as pd
    from nltk.tag import pos_tag
    
    words = list(set(brown.words()))
    pos_tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    HMM_POSTagger = POSTagger(tagged_sents, words, pos_tags)
    
    if len(sys.argv) >= 2 :
        s = sys.argv[1]
    else:
        s = input("Enter sentence to test : \n")
        
    test_sent = word_tokenize(s)
    O = [words.index(tok) if tok in words else len(words) for tok in test_sent]
    
    hmm_tagged_sent = HMM_POSTagger.tag(test_sent)
    hmm_tags = [wt[1] for wt in hmm_tagged_sent]
    
    print('\nObservation sequence and Optimal state sequence:\n', hmm_tagged_sent)
    
    rec_t = pos_tag(test_sent, tagset='universal')
    rec_t = [p[1] for p in rec_t]
    vi = pd.DataFrame(zip(test_sent, hmm_tags, rec_t, ['']*len(rec_t)), columns=['Tokens','Predicted POS tag','NLTK lib tagged','Mismatch'])
    vi['Mismatch'] = np.where(vi['Predicted POS tag']!=vi['NLTK lib tagged'], '●', '')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(vi)
