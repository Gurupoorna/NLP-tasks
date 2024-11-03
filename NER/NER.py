from sklearn.svm import SVC
import nltk
from nltk.corpus.reader import ConllCorpusReader
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import re
from pprint import pprint
import numpy as np
from itertools import chain
import pickle

with open('svmclassifier-5000.pkl','rb') as f:
    svmclassifier = pickle.load(f)
with open(f'red2v-5000.pkl', 'rb') as f:
    red2v = pickle.load(f)

def load_gazetteer_dict():
    with open('./gazetteer.txt') as f:
        lines = f.readlines()
        lines = [i[:-1] for i in lines]
        g_dict = defaultdict(set)
        for line in lines:
            tag, word = line.split()[0], (' ').join(line.split()[1:])
            g_dict[tag].add(word) #stores a set of words for each tag
    
    # print ('gazetteer dict sample: ',g_dict.keys())
    return g_dict
g_dict = load_gazetteer_dict()
g_dict.keys()

def word2features(i, wordseq):
    wi, postag = wordseq[i]
    features = defaultdict(lambda : -1)
    assert wordseq[0][0] == '<START>' and wordseq[-1][0] == '<STOP>', "<start> and <stop> tags missing"
    if wi == '<START>' or wi == '<STOP>':
        features.update({
            wi: True
        })
        return features
    
    alphabet = list('abcdefghijklmnopqrstuvwxyz')
    bow = dict()
    for char in alphabet:
        bow['$'+char] = 0
    for char in wi.lower():
        if char in alphabet:
            bow['$'+char] += 1

    features.update(bow)

    features.update({
        'iswialpha': wi.isalpha(),
        'iswititle': wi.istitle(),
        'iswiupper': wi.isupper(),
        'iswilower': wi.islower(),
        'iswidigit': wi.isdigit(),
        'iswinumeric': wi.isnumeric(),
        'Wishape': len(wi),
        'hasspecialchar': len([w for w in wi if w in list('~!@#$%^&*()_+-={}|:"<>?,./;\'[]')]) != 0,
        'isacronym': len(re.findall(r'(?:[A-Z]\.?){2,}' , wi)) > 0,
        'postag': postag,
    })
    if i>1:
        wiminus1, posminus1 = wordseq[i-1]
        features.update({
            'iswiminus1title': wiminus1.istitle(),
            'iswiminus1upper': wiminus1.isupper(),
            'iswiminus1lower': wiminus1.islower(),
            'postagminus1': posminus1,
        })
    elif i==1:
        features.update({
            'BOS':True,
            'iswiminus1title': -1,
            'iswiminus1upper': -1,
            'iswiminus1lower': -1,
            'postagminus1': -1,
        })
    if i<len(wordseq)-2:
        wiplus1, posplus1 = wordseq[i+1]
        features.update({
            'iswiplus1title': wiplus1.istitle(),
            'iswiplus1upper': wiplus1.isupper(),
            'iswiplus1lower': wiplus1.islower(),
            'postagplus1': posplus1,
        })
    elif i==len(wordseq)-2:
        features.update({
            'EOS':True,
            'iswiplus1title': -1,
            'iswiplus1upper': -1,
            'iswiplus1lower': -1,
            'postagplus1': -1,
        })

    features.update({
        "gaz" : -1, "gaz+1" : -1, "gaz-1" : -1, "gaz3" : -1,
        'gaztag-LOC' : -1, 'gaztag-PER' : -1, 'gaztag-ORG' : -1, 'gaztag-MISC' : -1,
        'gaz+1tag-LOC' : -1, 'gaz+1tag-PER' : -1, 'gaz+1tag-ORG' : -1, 'gaz+1tag-MISC' : -1,
        'gaz-1tag-LOC' : -1, 'gaz-1tag-PER' : -1, 'gaz-1tag-ORG' : -1, 'gaz-1tag-MISC' : -1,  
        'gaz3tag-LOC' : -1, 'gaz3tag-PER' : -1, 'gaz3tag-ORG' : -1, 'gaz3tag-MISC' : -1,  
    })

    if wi.isalnum():
        gaz = False
        gazplus1 = False
        gazminus1 = False
        gaz3 = False
        for k in g_dict.keys():
            if any(n in g_dict[k] for n in [wi.upper(), wi.lower(), wi.title(), wi.capitalize()]):
                gaz = True
                features.update({
                    'gaztag-'+str(k): 1,
                })
            if 'wiplus1' in locals():
                wd = wi+' '+wiplus1
                if any(name in g_dict[k] for name in [wd.upper(), wd.lower(), wd.title(), wd.capitalize()]):
                    gazplus1 = True
                    features.update({
                        'gaz+1tag-'+str(k): 1,
                    })
            if 'wiminus1' in locals() :
                wd = wiminus1+' '+wi
                if any(n in g_dict[k] for n in  [wd.upper(), wd.lower(), wd.title(), wd.capitalize()]):
                    gazminus1 = True
                    features.update({
                        'gaz-1tag-'+str(k): 1,
                    })
            if 'wiminus1' in locals() and 'wiplus1' in locals() :
                wd = wiminus1+' '+wi+' '+wiplus1 
                if any(n in g_dict[k] for n in  [wd.upper(), wd.lower(), wd.title(), wd.capitalize()]):
                    gaz3 = True
                    features.update({
                        'gaz3tag-'+str(k): 1,
                    })
        features.update({'gaz': gaz*100 , 'gaz+1': gazplus1*100 , 'gaz-1': gazminus1*100 , 'gaz3': gaz3*100})

    return features

def sent2features(sentence):
    assert isinstance(sentence, list) and isinstance(sentence[0], tuple) and isinstance(sentence[0][0], str), '`sentence` should be list of tuple of words and tags as str'
    xs = [None]*len(sentence)
    for i in range(len(sentence)):
        xs[i] = word2features(i, sentence)
    return xs

def feats2vects(features, test=False):
    if not test:
        tX = red2v.fit_transform(features)
    elif test:
        assert 'red2v' in globals() , 'no fit done earlier for dict to vect'
        tX = red2v.transform(features)
    return tX

def encode_ylabel(y):
    ty = [1 if label.startswith('B-') or label.startswith('I-') else 0 for label in y]
    return ty

def iob_sents2Xy(iob_sents, test=False) :
    sents_list = [[('<START>','<START>')]+[(w,p) for w, p, _ in wseq]+[('<STOP>','<STOP>')] for wseq in iob_sents]
    s2fs = list(map(sent2features, sents_list))
    Xs = list(chain.from_iterable(s2fs))
    ys = list(chain.from_iterable([['<START>']+[e for _, _, e in wseq]+['<STOP>'] for wseq in iob_sents]))
    y = encode_ylabel(ys)
    X = feats2vects(Xs, test=test)
    return np.array(X), np.array(y), {'sents_list':sents_list, 's2fs':s2fs, 'ys':ys}

def my_token_preps(user_sent):
    pattern = r'(\s|"|:|,|:|;|\'|!|\?|\(|\)|\.$)'
    s_l = list(filter(lambda x : ('').__ne__(x) and (' ').__ne__(x), re.split(pattern , user_sent)))
    if s_l[-1] != '.':
        s_l.append('.')
    return [('<START>','<START>')]+nltk.pos_tag(s_l)+[('<STOP>','<STOP>')]

if __name__ == '__main__':
    train = ConllCorpusReader('CoNLL-2003', 'eng.train', ['words', 'pos', 'ignore', 'chunk'])
    test = ConllCorpusReader('CoNLL-2003', 'eng.testa', ['words', 'pos', 'ignore', 'chunk'])
    red2v = DictVectorizer(sparse=False)
    
    all_train = train.iob_sents()
    all_test = test.iob_sents()
    fulltrainsize = len(all_train)
    fulltestsize = len(all_test)
    
    trainsize = 5000
    print('#'*60)
    print(f'Training size : {trainsize}')
    
    testsize = fulltestsize
    Xtrain, ytrain, propstr = iob_sents2Xy(all_train[:trainsize])
    print('X: ',Xtrain.shape, 'y: ',ytrain.shape)

    # Single
    # svmclassifier = SVC(probability=True)

    # svmclassifier.fit(Xtrain,ytrain)

    Xtest, ytest, propsts = iob_sents2Xy(all_test[:testsize], test=True)

    predictions = svmclassifier.predict(Xtest)

    score = svmclassifier.score(Xtest,ytest)

    # with open(f'svmclassifier-{trainsize}.pkl', 'wb') as f:
    #     pickle.dump(svmclassifier, f)
    # with open(f'red2v-{trainsize}.pkl', 'wb') as f:
    #     pickle.dump(red2v, f)

    print(f'{(predictions == ytest).sum()} correct out of {ytest.shape[0]} entities. '
        f'Accuracy = {(predictions == ytest).sum()/ytest.shape[0]}')
    print(f"Score on test set : {score}")
    print('Random Sample:')
    rr = 6890
    i=0
    for s in propsts['sents_list']:
        for w, p in s:
            if i>rr:
                print(f"{w:20s} - {p:5s} - {propsts['ys'][i]:6s} - {predictions[i] if predictions[i]==1 else ''}")
            i+=1
            if i-rr>40: break
        if i-rr>40: break

    user_sent = "Ludwig van Beethoven was a G.O.A.T from Germany who said, \"Nothing is more intolerable than to have to admit to yourself your own errors!\"\n --(1790s)"
    s_l = my_token_preps(user_sent)
    s2f = sent2features(s_l)
    x = feats2vects(s2f, test=True)
    user_nei = svmclassifier.predict(x)
    for w , e in zip(s_l, user_nei):
        print(f"{w:20s} - {e if e==1 else ''}")

    # viterbi
    # svmclassifier.predict_proba(x)