import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import pycrfsuite
from itertools import chain
# from collections import Counter

class CRFTagger():
    def __init__(self, ):
        sent_dataset = brown.tagged_sents(tagset='universal')
        self.sent_dataset = sent_dataset
        X_train = [self.sent2features(s) for s in sent_dataset]
        y_train = [self.sent2postags(s) for s in sent_dataset]

        self.trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            self.trainer.append(xseq, yseq)

        self.trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        self.trainer.train('browncorpus.crfsuite')

        self.tagger = pycrfsuite.Tagger()

        self.tagger.open('browncorpus.crfsuite')
        
        
    def tag(self, sent):
        return self.tagger.tag(self.sent2features([[tok] for tok in word_tokenize(sent)]))
        

    def test_tagger(self, tags_of_interest = ['NOUN','ADJ','ADP',]):
        train_sents, test_sents = train_test_split(self.sent_dataset, test_size=0.3, shuffle=True, random_state=100)
        X_train = [self.sent2features(s) for s in train_sents]
        y_train = [self.sent2postags(s) for s in train_sents]

        X_test = [self.sent2features(s) for s in test_sents]
        y_test = [self.sent2postags(s) for s in test_sents]

        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })

        trainer.train('browncorpus_test_set.crfsuite')

        tagger = pycrfsuite.Tagger()
        tagger.open('browncorpus_test_set.crfsuite')

        y_pred = [tagger.tag(xseq) for xseq in X_test]

        print(self.get_classification_report(y_test, y_pred))

        y_true_combined = list(chain.from_iterable(y_test))
        y_pred_combined = list(chain.from_iterable(y_pred))

        import pandas as pd
        df = pd.DataFrame(
            confusion_matrix(y_true_combined, y_pred_combined, labels=tags_of_interest),
            columns=tags_of_interest,
            index=tags_of_interest
        )
        return df

    def get_classification_report(self, y_true, y_pred):
        lb = LabelBinarizer()
        y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
        y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

        tagset = lb.classes_
        return classification_report(
            y_true_combined,
            y_pred_combined,
            labels = range(len(tagset)),
            target_names = tagset,
        )
    
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


    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2postags(self, sent):
        return [postag for token, postag in sent]

    def sent2tokens(self, sent):
        return [token for token, postag in sent]