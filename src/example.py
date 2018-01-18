from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

clf = svm.SVC(gamma=0.001, C=100., probability=True)
clf.fit(digits.data[:-1], digits.target[:-1])

# print(clf.predict(digits.data[-1:]))
