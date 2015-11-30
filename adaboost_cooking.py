# Standard Imports
import pandas as pd
import numpy as np

# Performance
from time import time


# Helper
from sklearn.cross_validation import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.externals.six.moves import zip

import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Load in the Data
# train = pd.read_json('data/train.json')
train = pd.read_json('data/train1.json')

# Extract the Unique Ingredients
words = [' '.join(item) for item in train.ingredients]

# Construct the Bag of Words
vec = CountVectorizer(max_features=2000)

bag_of_words = vec.fit(words).transform(words).toarray()

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=10),
    n_estimators=500,
    learning_rate=1.5,
    algorithm="SAMME.R")

start = time()
ada_clf.fit(bag_of_words, train.cuisine)
print("adaboost classifier training in %.2f" % (time() - start))

# use cross-validation to estimate accuracy
#start = time()
#train_pred = cross_val_predict(ada_clf, bag_of_words, train.cuisine, cv=2)
#print("adaboost evaluation finished in %.2f" % (time() - start))

#print("Estimated accuracy using cross-validation: " , accuracy_score(train.cuisine, train_pred))


# use rest of labelled training data to check accuracy score (for plotting)
test = pd.read_json('data/train2.json')
test_words = [' '.join(item) for item in test.ingredients]
test_bag = vec.transform(test_words).toarray()
test_errors = []
for test_predict in ada_clf.staged_predict(test_bag):
    test_errors.append( 1. - accuracy_score(test_predict, test.cuisine) )

plt.figure(figsize=(15, 5))

plt.plot(range(1, len(ada_clf) + 1), test_errors)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')
plt.show()

# Load in Testing Data
test = pd.read_json('data/test.json')

# Create test Bag of Words
test_words = [' '.join(item) for item in test.ingredients]
test_bag = vec.transform(test_words).toarray()
result = ada_clf.predict(test_bag)

output = pd.DataFrame(data={"id":test.id, "cuisine":result})

output.to_csv("data/submission.csv", index=False, quoting=3)
         
