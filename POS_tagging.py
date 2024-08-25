import nltk
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('brown',quiet=True)
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import numpy as np
from numba import njit, prange

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from viterbi import viterbi_log

from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO,
        format='\n%(asctime)s [%(levelname)s] %(name)s - %(message)s')

class POSTagger():
    def __init__(self, tagged_sentences:list[list[tuple[str,str]]], 
                 words=None, pos_tags=None, A=None, B=None, Pi=None):
        self._logger = logging.getLogger('POSTagger')
        self._logger.info('Initializing Hidden Markov Model.')
        if not words:
            words = list(set([wt[0] for s in tqdm(tagged_sentences, desc='Getting all words ') for wt in s]))
        if not pos_tags: words = list(set([wt[1] for s in tagged_sentences for wt in s]))
        self._logger.info(f'Universal tagset used : {pos_tags}')
        self.words = words
        self.pos_tags = pos_tags
        self._no_words = len(self.words)
        self._no_tags = len(self.pos_tags)
        self._X_id_map = {v: i for i, v in enumerate(self.words)}
        self._Y_id_map = {v: i for i, v in enumerate(self.pos_tags)}
        if not A or not B or not Pi:
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
        O = [words.index(tok) if tok in words else self._no_words for tok in toks]
        optm_seq, _, _ = viterbi_log(O, self._transProb, self._iniProb, self._emissProb)
        optm_seq = [pos_tags[t] for t in optm_seq]
        test_tagged_sent  = list(zip(toks,optm_seq))
        return test_tagged_sent

    def batch_tag(self, X_test):
        if not isinstance(X_test,np.ndarray):
            X_test = self._to_numpy(X_test, give_out=True)
        self._logger.info('Evaluating optimal state sequence using log-viterbi algorithm...')
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
    
if __name__ == '__main__':
    words = list(set(brown.words()))
    pos_tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    HMM_POSTagger = POSTagger(tagged_sents, words, pos_tags)

    print("Testing for some arbitrary sentence from corpus")
    r = 1028              # sample number to test
    test_sent = []  
    test_tags = []
    for word, tag in tagged_sents[r]:
        test_sent.append(word)
        test_tags.append(tag)
        
    hmm_tagged_sent = HMM_POSTagger.tag(test_sent)
    hmm_tags = [wt[1] for wt in hmm_tagged_sent]
    tag_matching = np.asarray(test_tags)==np.asarray(hmm_tags)
    
    print('Observation sequence:   O  = ', test_sent)
    print('Correct state sequence: S* = ', test_tags)
    print('Optimal state sequence: S  = ', hmm_tags)
    print("Do they match ? : ", tag_matching.all())
    
    import pandas as pd
    test_result_df = pd.DataFrame(zip(test_sent, hmm_tags, test_tags, np.where(tag_matching,'','●')), columns=['Tokens','Predicted POS tag','Gold tag','Errors'])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(test_result_df)
        
    #########################    K-FOLD    #########################
    # Define number of folds
    kf = KFold(n_splits=5, shuffle=True, random_state=89)
    
    ST = HMM_POSTagger.get_ST()
    
    accuracies = []
    conf_matrices = []
    print('Conducting K-fold cross-validation :\n')
    for train_index, test_index in tqdm(kf.split(ST), desc='cross-validation '):
        # Split data into training and testing sets
        train_ST = ST[train_index]
        test_ST =  ST[test_index]
        test_ST_o, test_ST_q = np.moveaxis(test_ST,-1,0)
    
        # Testing
        pred = HMM_POSTagger.batch_tag(test_ST_o)
        
        assert ((pred == -1) == (test_ST_q == -1)).all()
        conf_matx = confusion_matrix(pred[pred!=-1].ravel(),test_ST_q[test_ST_q!=-1].ravel())
        
        # Per label accuracy which i think turns out same as precision
        prec_s = conf_matx.sum(axis=0)
        prec_s[prec_s==0] = 1 # handling division by zero
        precis = conf_matx.diagonal() / prec_s
        
        accuracies.append(precis)
        conf_matrices.append(conf_matx)
    
    per_tag_acc = np.array(accuracies).mean(axis=0)
    per_tag_acc = np.asarray(list(zip(pos_tags, per_tag_acc)))
    with np.printoptions(precision=4,suppress=True):
        print(f'Per tag accuracies :\n{per_tag_acc}')
    # IF to print c-matrix in of every fold
    # print('\nConfusion matrices from every fold :')
    # for i in range(len(conf_matrices)):
    #     print(f'{i+1}-fold :\n{conf_matrices[i]}')
    ######################    K-FOLD-ENDS    #######################
    
    ######################     USER INPUT    #######################
    from nltk.tokenize import word_tokenize
    s = input("Enter sentence to test : \n")
    test_sent = word_tokenize(s)
    O = [words.index(tok) if tok in words else len(words) for tok in test_sent]
    
    hmm_tagged_sent = HMM_POSTagger.tag(test_sent)
    hmm_tags = [wt[1] for wt in hmm_tagged_sent]
    
    print('\nObservation sequence and Optimal state sequence:\n', hmm_tagged_sent)
    
    from nltk.tag import pos_tag
    rec_t = pos_tag(test_sent, tagset='universal')
    rec_t = [p[1] for p in rec_t]
    vi = pd.DataFrame(zip(test_sent, hmm_tags, rec_t, ['']*len(rec_t)), columns=['Tokens','Predicted POS tag','NLTK lib tagged','Mismatch'])
    vi['Mismatch'] = np.where(vi['Predicted POS tag']!=vi['NLTK lib tagged'], '●', '')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(vi)