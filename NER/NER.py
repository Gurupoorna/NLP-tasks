from sklearn.svm import LinearSVC, SVC
import nltk
from nltk.corpus.reader import ConllCorpusReader

train = ConllCorpusReader('CoNLL-2003', 'eng.train', ['words', 'pos', 'ignore', 'chunk'])
test = ConllCorpusReader('CoNLL-2003', 'eng.testb', ['words', 'pos', 'ignore', 'chunk'])


nerclassfy = LinearSVC()

# nerclassfy.fit(X,y)