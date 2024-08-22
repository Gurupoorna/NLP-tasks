from nltk.corpus import brown
import numpy as np
from numba import njit
from viterbi import viterbi_log

def to_numpy(XY:list[list[tuple[str,str]]], X_name:list, Y_name:list, to_sort=False):
    X_id_map = {v: i for i, v in enumerate(X_name)}
    Y_id_map = {v: i for i, v in enumerate(Y_name)}
    if to_sort: XY = sorted(XY,key=len)
        
    n = len(XY)
    m = max(len(s) for s in XY)
    ST = np.zeros((n, m, 2), dtype=np.int32)-1

    for i, s in enumerate(XY):
        for j, w in enumerate(s):
            ST[i, j] = np.array([X_id_map[w[0]], Y_id_map[w[1]]])

    return ST

@njit
def get_prob(stB:np.ndarray, no_words:int, no_tags:int, eps:float=0.00001):
    A = np.zeros((no_tags,no_tags), dtype=np.float64)
    B = np.zeros((no_tags,no_words), dtype=np.float64)
    Pi = np.zeros(no_tags, dtype=np.float64)

    m, n, _ = stB.shape
    
    for i in range(m):
        if stB[i,0,0] == -1 : break
        Pi[stB[i,0,1]] += 1
        for j in range(n-1):
            wnth = stB[i,j,0]
            nth, nnth = stB[i,j:j+2,1]
            B[nth, wnth] += 1
            if nnth == -1 : break
            A[nth,nnth] += 1
        if nnth != -1 : B[stB[i,n-1,1], stB[i,n-1,0]] += 1
    A /= np.sum(A,axis=1).reshape(-1,1) + eps
    B /= np.sum(B,axis=1).reshape(-1,1) + eps
    Pi /= np.sum(Pi) + eps
    return A, B, Pi

if __name__ == '__main__':
    words = list(set(brown.words()))
    pos_tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    ST = to_numpy(tagged_sents, words, pos_tags)
    A, B, Pi = get_prob(ST, len(words), len(pos_tags))

    r = 1028              # sample number to test
    test_sent = []
    test_tags = []
    for word, tag in tagged_sents[r]:
        test_sent.append(word)
        test_tags.append(tag)
    O = [words.index(word) for word in test_sent] # producing observation states (words) in terms of its word index
        
    optim_state_seq, log_prob_trellis, backtrack_matrix = viterbi_log(O,A,Pi,B)
    
    optim_state_seq = [pos_tags[t] for t in optim_state_seq]
    
    print('Observation sequence:   O  = ', test_sent)
    print('Correct state sequence: S* = ', test_tags)
    print('Optimal state sequence: S  = ', optim_state_seq)
    print("Do they match ? : ", test_tags==optim_state_seq)
    
    from pandas import DataFrame
    test_result_df = DataFrame(index=test_sent,columns=['Correct','Guessed'],data=zip(test_tags,optim_state_seq)).T
    print(test_result_df.iloc[:,(test_result_df.nunique()!=1).values])