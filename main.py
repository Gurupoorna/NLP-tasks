import nltk
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from POSTagger import POSTagger

# Function to initialize the HMM tagger
def initialize_hmm_tagger():
    words = list(set(brown.words()))
    pos_tags = ['VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.']
    tagged_sents = list(brown.tagged_sents(tagset='universal'))

    # Initialize the POSTagger (HMM)
    hmm_tagger = POSTagger(tagged_sents, words, pos_tags)
    return hmm_tagger, words

# Function to perform POS tagging on a given sentence
def pos_tag_sentence(hmm_tagger, sentence, words):
    # Tokenize the sentence
    test_sent = word_tokenize(sentence)

    # Get the HMM-tagged sentence
    hmm_tagged_sent = hmm_tagger.tag(test_sent)
    hmm_tags = [wt[1] for wt in hmm_tagged_sent]

    # Return the tagged sentence
    return hmm_tagged_sent

# Example function for performing K-fold cross-validation (not relevant for UI)
def perform_k_fold_validation(hmm_tagger, words, pos_tags):
    from sklearn.model_selection import KFold
    import numpy as np

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=89)
    ST = hmm_tagger.get_ST()

    print('Conducting K-fold cross-validation:')
    accuracies, recalls, fscores, conf_matrices = do_kfold(kf.split(ST), k, words, pos_tags, ST)

    per_tag_acc = np.array(accuracies).mean(axis=0)
    per_tag_recl = np.array(recalls).mean(axis=0)
    per_tag_fscore = np.array(fscores).mean(axis=0)

    with np.printoptions(precision=4, suppress=True):
        print(f'Per tag accuracies :\n{list(zip(pos_tags, per_tag_acc))}')
        print(f'Per tag recall :\n{list(zip(pos_tags, per_tag_recl))}')
        print(f'F-scores :\n{list(zip(pos_tags, per_tag_fscore))}')

    for i, matrix in enumerate(conf_matrices):
        print(f'{i+1}-fold C-matrix:\n{matrix}')

    