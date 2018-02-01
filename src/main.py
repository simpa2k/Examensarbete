from sklearn.naive_bayes import MultinomialNB

from src.featurizers.bow_featurize import featurize
from src.loaders.file_loader import create_file_loader

ALLOWED_EXTENSIONS = [".java"]

path_to_data = "../data"
path_to_corpus = "../output/corpus.mm"

features = featurize(create_file_loader(path_to_data, ALLOWED_EXTENSIONS), path_to_corpus)

clf = MultinomialNB()
clf.fit(features["data"], features["target"])

answer = clf.predict(features["data"][0:])
print("Guess was: ", answer)
