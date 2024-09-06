import nltk
nltk.download('universal_tagset', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import word_tokenize
from HMM import initialize_hmm_tagger, perform_validations
import sys
import os
import pickle
import json
if __name__ == '__main__' :
    seed=571
    hmm_tagger, words, pos_tags = initialize_hmm_tagger()
    if not os.path.exists('hmm_probs.npz'):
        hmm_tagger.save_prob_np('hmm_probs')
    if len(sys.argv) >= 2 :
        hmm_tagged_sent = hmm_tagger.tag(sys.argv[1])
        print('\nObservation sequence and Optimal state sequence:\n', hmm_tagged_sent)
    else:
        pass # s = input("Enter sentence to test : \n")
    print('Conducting K-fold cross-validation:')
    results = perform_validations(hmm_tagger, words, pos_tags, k=5, random_state=seed)
    print('Total Accuracy :', results['total_accuracy'])
    print('Total Recall : ', results['total_recall'])
    print('Total F-score :', results['total_fscore'])
    results['seed'] = seed
    with open('hmm_results.pkl', 'wb') as p:
        pickle.dump(results, p)
    with open('scores.txt','a') as f:
        f.write(json.dumps(results))