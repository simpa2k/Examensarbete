from sklearn.naive_bayes import MultinomialNB
from src.featurize import featurize
from sklearn.naive_bayes import MultinomialNB

from src.featurize import featurize

path_to_data = "/Users/simonolofsson/programmering/c++/bayes"
path_to_corpus = '../output/corpus.mm'

features = featurize(path_to_data, path_to_corpus)

# clf = svm.SVC(gamma=0.001, C=100., probability=True)
# clf.fit(features['data'][:-1], features['target'][:-1])
clf = MultinomialNB()
clf.fit(features['data'][:-1], features['target'][:-1])

answer = clf.predict(features['data'][-1:])
print("Guess was: ", answer)
