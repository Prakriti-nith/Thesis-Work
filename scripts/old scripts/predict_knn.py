# import pandas as pd
import csv
from sklearn.neighbors import KNeighborsClassifier 

train_X = []
test_X = []
train_Y = []
test_Y = []
with open('training_data.csv', 'r') as f:
  reader = csv.reader(f)
  train_X = list(reader)

print(len(train_X))
print(len(train_X[0]))


with open('testing_data.csv', 'r') as f:
  reader = csv.reader(f)
  test_X = list(reader)

with open('training_data_labels.csv', 'r') as f:
  reader = csv.reader(f)
  temp_train_Y = list(reader)
  train_Y = [label for sublist in temp_train_Y for label in sublist]

with open('testing_data_labels.csv', 'r') as f:
  reader = csv.reader(f)
  temp_test_Y = list(reader)
  test_Y = [label for sublist in temp_test_Y for label in sublist]

knn = KNeighborsClassifier(n_neighbors = 5).fit(train_X, train_Y) 
# model accuracy for test_X   
accuracy = knn.score(test_X, test_Y) 

# 43.1%
print(accuracy)
