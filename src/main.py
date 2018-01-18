from sklearn.naive_bayes import MultinomialNB

from src.featurizers.vectorizing_featurize import featurize
from src.loaders.file_loader import create_file_loader

ALLOWED_EXTENSIONS = [".cpp", ".h"]

path_to_data = "/home/simon/programmering/c++"
path_to_corpus = '../output/corpus.mm'

features = featurize(create_file_loader(path_to_data, ALLOWED_EXTENSIONS), path_to_corpus)

clf = MultinomialNB()
clf.fit(features['data'][:-1], features['target'][:-1])

answer = clf.predict(features['data'][-1:])
print("Guess was: ", answer)
