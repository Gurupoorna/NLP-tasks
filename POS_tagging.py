from nltk.corpus import treebank
import numpy as np
from viterbi import viterbi_log

def make_prob_matrices(tagged_sentences,words,pos_tags):
    """ Contructs probability matrices.        
    
    Args:
        tagged_sentences (List[List[Tuple(str,str)],...] List of sentences, each of which is POS tagged as [(<WORD>, <TAG>),...]
        words (List[str]): List of all unique words. The index in this list will be later used for learning.
        pos_tags (List[str]): List of all unique POS tags occuring in corpus. Likewise, index will be used for learning

    Returns:
        A:  Transition probability matrix where A[i,j] := P(pos_tags[j] | pos_tag[i]), shape(# tags,# tags)
        B:  Probability matrix where B[i,j] := P(words[j] | pos_tags[i]), shape(# words,# tags)
        Pi: Initial tag probability where Pi[i] := P(words[i] | ^), shape(# tags)
    """
    A = np.zeros((len(pos_tags),len(pos_tags)),dtype=np.float64)
    B = np.zeros((len(pos_tags),len(words)),dtype=np.float64)
    Pi = np.zeros(len(pos_tags),dtype=np.float64)
    for sentence in tagged_sentences:
        Pi[pos_tags.index(sentence[0][1])] += 1
        for n in range(len(sentence)-1):
            nth, nnth = sentence[n:n+2]
            A[pos_tags.index(nth[1]),pos_tags.index(nnth[1])] += 1
            B[pos_tags.index(nth[1]), words.index(nth[0])] += 1
        B[pos_tags.index(sentence[-1][1]), words.index(sentence[-1][0])] += 1
    A /= np.sum(A,axis=1).reshape(-1,1)
    B /= np.sum(B,axis=1).reshape(-1,1)
    Pi /= np.sum(Pi)
    return A, B, Pi


if __name__ == '__main__':
    words = list(set(treebank.words())) # unique list of words
    pos_tags = list(set(pair[1] for pair in treebank.tagged_words())) # extract all unique tags from corpus
    
    A, B, Pi = make_prob_matrices(treebank.tagged_sents(), words, pos_tags)

    r = 303              # sample number to test
    test_sent = treebank.sents()[r]
    O = [words.index(word) for word in test_sent] # producing observation states (words) in terms of its word index
    test_tagged = [tag for _, tag in treebank.tagged_sents()[r]]
    correct_tag_seq = np.array([pos_tags.index(pair[1]) for pair in treebank.tagged_sents()[r]])

    opt_state_seq, log_prob_trellis, backtrack_matrix = viterbi_log(A,Pi,B,O)


    # The following was to check if it was working or not. These parts need to be better done.
    print('Observation sequence:   O  = ', O)
    print('Optimal state sequence: S  = ', opt_state_seq)
    print('Correct state sequence: S* = ', correct_tag_seq)
    print("Do they match : ", correct_tag_seq==opt_state_seq)

    guessed_tags = [pos_tags[i] for i in opt_state_seq]
    from pandas import DataFrame
    test_result_df = DataFrame(index=test_sent,columns=['Correct','Guessed'],data=zip(test_tagged,guessed_tags)).T
    print('The sentence : ', test_sent)
    print(test_result_df.iloc[:,(test_result_df.nunique()!=1).values])