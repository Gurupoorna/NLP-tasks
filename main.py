import nltk
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from POSTagger import POSTagger

# Function to initialize the HMM tagger
def initialize_hmm_tagger(A=None, B=None, Pi=None):
    words = sorted(list(set(brown.words())))
    pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', '.', 'X']

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
def perform_k_fold_validation(hmm_tagger, words, pos_tags):
    from sklearn.model_selection import KFold
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    import numpy as np
    import pandas as pd

    k = 2
    kf = KFold(n_splits=k, shuffle=True, random_state=89)
    ST = hmm_tagger.get_ST()

    # Split data into training and testing sets
    no_pos_tags = len(pos_tags)
    accuracies = np.empty((k, no_pos_tags), dtype=np.float64)
    recalls    = np.empty((k, no_pos_tags), dtype=np.float64)
    fscores    = np.empty((k, no_pos_tags), dtype=np.float64)
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
        accuracies[i] = precision_s
        recalls[i]    = recall_s
        fscores[i]    = fscore_s
        c_matrices[i] = conf_matx
        i+=1
    
    per_tag_acc = np.array(accuracies).mean(axis=0)
    per_tag_recl = np.array(recalls).mean(axis=0)
    per_tag_fscore = np.array(fscores).mean(axis=0)
    metrics = pd.DataFrame(zip(pos_tags, per_tag_acc, per_tag_recl, per_tag_fscore), columns=['Tags', 'Accuracy', 'Recall', 'F-score'])
    print(metrics)
    for i, matrix in enumerate(c_matrices):
        print(f'{i+1}-fold C-matrix:\n{matrix}')


if __name__ == '__main__' :
    import sys
    hmm_tagger, words, pos_tags = initialize_hmm_tagger()
    hmm_tagger.save_prob_np('hmm_probs')
    if len(sys.argv) >= 2 :
        s = sys.argv[1]
        test_sent = word_tokenize(s)
        O = [words.index(tok) if tok in words else len(words) for tok in test_sent]
        hmm_tagged_sent = hmm_tagger.tag(test_sent)
        hmm_tags = [wt[1] for wt in hmm_tagged_sent]
        print('\nObservation sequence and Optimal state sequence:\n', hmm_tagged_sent)
    else:
        pass # s = input("Enter sentence to test : \n")
    print('Conducting K-fold cross-validation:')
    perform_k_fold_validation(hmm_tagger, words, pos_tags)