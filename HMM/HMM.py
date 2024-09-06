from nltk.tokenize import word_tokenize
from nltk.corpus import brown
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

        if words is None:
            words = sorted(list(set([wt[0] for s in tqdm(tagged_sentences, desc='Getting all words') for wt in s])))
        if pos_tags is None:
            pos_tags = list(set([wt[1] for s in tqdm(tagged_sentences, desc='Getting all tags') for wt in s]))
            pos_tags.remove('X')
            pos_tags.append('X')

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
        elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray) and isinstance(Pi, np.ndarray):
            self._transProb = A
            self._emissProb = B
            self._iniProb = Pi
            self._logger.info('Assigning pre-computed probability matrices.')
        else:
            self._to_numpy(tagged_sentences)
            self._logger.info('Calculating probability matrices...')
            self._transProb, self._emissProb, self._iniProb = self._get_prob(self.words, self.pos_tags, self._ST)
    
    def save_prob_np(self, name):
        np.savez_compressed(name, A=self._transProb, B=self._emissProb, Pi=self._iniProb)

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
    def _get_prob(words, pos_tags, _ST, eps=np.finfo(float).tiny):
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


# Function to initialize the HMM tagger
def initialize_hmm_tagger(A=None, B=None, Pi=None):
    words = sorted(list(set(brown.words())))
    # pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', '.', 'X']
    pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', '.', 'PRT', 'X']

    # Initialize the POSTagger (HMM)
    if A is None and B is None and Pi is None :
        tagged_sents = list(brown.tagged_sents(tagset='universal'))
        hmm_tagger = POSTagger(tagged_sents, words, pos_tags)
    else :
        hmm_tagger = POSTagger(words=words, pos_tags=pos_tags, A=A, B=B, Pi=Pi)
    return hmm_tagger, words, pos_tags

# Function to perform POS tagging on a given sentence
def pos_tag_sentence(hmm_tagger, sentence, words):
    # Tokenize the sentence
    test_sent = word_tokenize(sentence)

    # Get the HMM-tagged sentence
    hmm_tagged_sent = hmm_tagger.tag(test_sent)

    # Return the tagged sentence
    return hmm_tagged_sent

# Function for performing K-fold cross-validation
def perform_validations(hmm_tagger, words, pos_tags, k=2, betas=[1,0.5,2], random_state=123):
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from pandas import DataFrame

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    ST = hmm_tagger.get_ST()

    # Split data into training and testing sets
    no_pos_tags = len(pos_tags)
    accuracies = np.empty((k, no_pos_tags), dtype=np.float64)
    recalls    = np.empty((k, no_pos_tags), dtype=np.float64)
    fscores    = np.empty((k, no_pos_tags), dtype=np.float64)
    fbetascores = np.empty((k, no_pos_tags, len(betas)), dtype=np.float64)
    c_matrices = np.empty((k, no_pos_tags, no_pos_tags), dtype=np.int_)
    i = 0
    for train_index, test_index in kf.split(ST):
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
        fbetascores_s = []
        for beta in betas:
            _, _, fbscore_s, _ = precision_recall_fscore_support(
            y_true, y_pred, 
            beta=beta, 
            labels=range(no_pos_tags)
        )
            fbetascores_s.append(fbscore_s)

        accuracies[i] = precision_s
        recalls[i]    = recall_s
        fscores[i]    = fscore_s
        c_matrices[i] = conf_matx
        fbetascores[i] = np.array(fbetascores_s).reshape(len(pos_tags), len(betas))
        i+=1
    
    per_tag_acc    = accuracies.mean(axis=0)
    per_tag_recl   = recalls.mean(axis=0)
    per_tag_fscore = fscores.mean(axis=0)
    per_tag_fbetascores = fbetascores.mean(axis=0)
    
    total_accuracy = per_tag_acc.mean()
    total_recall = per_tag_recl.mean()
    total_fscore = per_tag_fscore.mean()
    metrics = DataFrame(zip(pos_tags, per_tag_acc, per_tag_recl, per_tag_fscore, *per_tag_fbetascores.T), columns=['Tags', 'Accuracy', 'Recall', 'F-score']+[f'F-beta={b}-score' for b in betas])
    print(metrics)
    for i, matrix in enumerate(c_matrices):
        print(f'{i+1}-fold C-matrix:\n{matrix}')
    
    return {
        'accuracies': accuracies,
        'recalls': recalls,
        'fscores': fscores,
        'per_tag_acc': per_tag_acc,
        'per_tag_recl': per_tag_recl,
        'per_tag_fscore': per_tag_fscore,
        'per_tag_fbetascores': per_tag_fbetascores,
        'metrics': metrics,
        'c_matrices': c_matrices,
        'total_accuracy': total_accuracy,
        'total_recall': total_recall,
        'total_fscore': total_fscore
    }