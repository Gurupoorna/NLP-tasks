import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pycrfsuite
from itertools import chain
import logging

class CRFTagger():
    sent_dataset = brown.tagged_sents(tagset='universal')
    logging.basicConfig(level=logging.INFO, format='\n%(asctime)s [%(levelname)s] %(name)s - %(message)s')
    logger = logging.getLogger('CRFTagger')
    def __init__(self, name, from_saved=False):# train_sents=None, test_sents=None):
        self.name = name
        if from_saved:
            self.logger.info(f'Fetching saved {name}.crfsuite')
            self.tagger = pycrfsuite.Tagger()
            self.tagger.open(f'{name}.crfsuite')
            return 
        self.sent_dataset = __class__.sent_dataset
        X_train = [self.sent2features(s) for s in self.sent_dataset]
        y_train = [self.sent2postags(s) for s in self.sent_dataset]

        self.trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            self.trainer.append(xseq, yseq)

        self.trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            'feature.possible_transitions': True # include transitions that are possible, but not observed
        })
        self.logger.info('Start CRF training')
        self.trainer.train(f'{name}.crfsuite')
        self.logger.info('Finished training')
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(f'{name}.crfsuite')
        
    def tag(self, sent:str, tup=False):
        tokens = word_tokenize(sent)
        pred_tags = self.tagger.tag(self.sent2features([[tok] for tok in tokens]))
        if tup:
            return list(zip(tokens,pred_tags))
        return pred_tags
    
    @classmethod
    def test_tagger(cls, name, train_sents=None, test_sents=None):
        if train_sents is None and test_sents is None : 
            train_sents, test_sents = train_test_split(__class__.sent_dataset, test_size=0.3, shuffle=True, random_state=100)
        X_train = [cls.sent2features(s) for s in train_sents]
        y_train = [cls.sent2postags(s) for s in train_sents]
        X_test = [cls.sent2features(s) for s in test_sents]
        y_test = [cls.sent2postags(s) for s in test_sents]
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier
            'feature.possible_transitions': True # include transitions that are possible, but not observed
        })
        cls.logger.info('Test crf training')
        trainer.train(f'{name}_test_set.crfsuite')
        
        tagger = pycrfsuite.Tagger()
        tagger.open(f'{name}_test_set.crfsuite')
        y_pred = [tagger.tag(xseq) for xseq in X_test]
        cls.logger.info('Returning results')
        return cls.get_classification_report(y_test, y_pred)
        
    @classmethod
    def get_classification_report(cls, y_true, y_pred):
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = lb.classes_
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = range(len(tagset)),
            target_names = tagset,
            output_dict = True,
        )
    @classmethod
    def word2features(self, sent, i):
        word = sent[i][0]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
        ]
        if i > 0:
            word1 = sent[i-1][0]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
            ])
        else:
            features.append('BOS')
            
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
            ])
        else:
            features.append('EOS')
                    
        return features
    @classmethod
    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]
    @classmethod
    def sent2postags(self, sent):
        return [postag for token, postag in sent]
    @classmethod
    def sent2tokens(self, sent):
        return [token for token, postag in sent]
    

# Function to initialize the CRF tagger
def initialize_crf_tagger(name):
    import os
    if os.path.exists(name+'.crfsuite') :
        crf_tagger = CRFTagger(name, from_saved=True)
    else :
        # Initialize the CRFTagger
        crf_tagger = CRFTagger(name)
    return crf_tagger

# Function to perform POS tagging on a given sentence
def pos_tag_sentence(crf_tagger:CRFTagger, sentence):
    # Get the CRF-tagged sentence
    crf_tagged_sent = crf_tagger.tag(sentence, tup=True)
    # Return the tagged sentence
    return crf_tagged_sent

# Function for performing K-fold cross-validation
def perform_validations(k=2, random_state=481):
    from sklearn.model_selection import KFold
    import numpy as np
    from pandas import DataFrame

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    
    # Split data into training and testing sets
    all_sents = np.asarray(CRFTagger.sent_dataset, dtype=object)
    k_reports = []
    for train_index, test_index in kf.split(all_sents):
        train_sents = all_sents[train_index]
        test_sents =  all_sents[test_index]
        # Testing
        k_reports.append(CRFTagger.test_tagger('crf_k', train_sents, test_sents))
    k_metrics = np.array([dfi.T.to_numpy()[:,:-1] for dfi in map(lambda td: DataFrame.from_dict(td), k_reports)])
    over_all_metrics = k_metrics.mean(0)
    df = DataFrame.from_dict(k_reports[0]).T.drop('support',axis=1)
    return DataFrame(data=over_all_metrics , columns=df.columns , index=df.index)