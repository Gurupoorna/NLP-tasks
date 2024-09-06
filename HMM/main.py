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
    imp_results = {
        'seed': seed,
        'total_accuracy': results['total_accuracy'],
        'total_recall': results['total_recall'],
        'total_fscore': results['total_fscore'],
        'per_tag_acc': results['per_tag_acc'].tolist(),
        'per_tag_recl': results['per_tag_recl'].tolist(),
        'per_tag_fscore': results['per_tag_fscore'].tolist(),
        'per_tag_fbetascores': results['per_tag_fbetascores'].tolist(),
    }
    results['seed'] = seed
    with open('hmm_results.pkl', 'wb') as p:
        pickle.dump(results, p)
    with open('scores.txt','a+') as f:
        f.write(json.dumps(imp_results, indent=4))
        f.write('\n\n')
        