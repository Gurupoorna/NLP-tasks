import nltk
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown',quiet=True)
nltk.download('punkt', quiet=True)

from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from POSTagger import POSTagger, do_kfold

    
if __name__ == '__main__':
    print('Enter main')
    words = list(set(brown.words()))
    pos_tags = ['VERB','NOUN','PRON','ADJ','ADV','ADP','CONJ','DET','NUM','PRT','X','.']
    tagged_sents = list(brown.tagged_sents(tagset='universal'))
    
    HMM_POSTagger = POSTagger(tagged_sents, words, pos_tags)

    #####################  SAMPLE OUTPUT  #####################
    # print("Testing for some arbitrary sentence from corpus")
    # r = 1028              # sample number to test
    # test_sent = []  
    # test_tags = []
    # for word, tag in tagged_sents[r]:
    #     test_sent.append(word)
    #     test_tags.append(tag)
        
    # hmm_tagged_sent = HMM_POSTagger.tag(test_sent)
    # hmm_tags = [wt[1] for wt in hmm_tagged_sent]
    # tag_matching = np.asarray(test_tags)==np.asarray(hmm_tags)
    
    # print('Observation sequence:   O  = ', test_sent)
    # print('Correct state sequence: S* = ', test_tags)
    # print('Optimal state sequence: S  = ', hmm_tags)
    # print("Do they match ? : ", tag_matching.all())
    
    # test_result_df = pd.DataFrame(zip(test_sent, hmm_tags, test_tags, np.where(tag_matching,'','●')), columns=['Tokens','Predicted POS tag','Gold tag','Errors'])
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(test_result_df)
    ###################  SAMPLE OUTPUT ENDS  ###################
    
    
    #########################    K-FOLD    #########################
    # Define number of folds
    k=5
    kf = KFold(n_splits=k, shuffle=True, random_state=89)
    ST = HMM_POSTagger.get_ST()
    print('Conducting K-fold cross-validation :')
    accuracies, recalls, fscores, conf_matrices = do_kfold(kf.split(ST), k, words, pos_tags, ST)

    per_tag_acc = np.array(accuracies).mean(axis=0)
    per_tag_acc = np.asarray(list(zip(pos_tags, per_tag_acc)))
    per_tag_recl = np.array(recalls).mean(axis=0)
    per_tag_recl = np.asarray(list(zip(pos_tags, per_tag_recl)))
    per_tag_fscore = np.array(fscores).mean(axis=0)
    per_tag_fscore = np.asarray(list(zip(pos_tags, per_tag_fscore)))
    
    with np.printoptions(precision=4,suppress=True):
        print(f'Per tag accuracies :\n{per_tag_acc}')
        print(f'Per tag recall :\n{per_tag_recl}')
        print(f'F-scores :\n{per_tag_fscore}')
    # IF to print c-matrix in of every fold
    print('\nConfusion matrices from every fold :')
    for i in range(len(conf_matrices)):
        print(f'{i+1}-fold C-matrix:\n{conf_matrices[i]}')
    ######################    K-FOLD-ENDS    #######################
    
    
    ######################     USER INPUT    #######################
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
    ######################     USER INPUT ENDS    #######################