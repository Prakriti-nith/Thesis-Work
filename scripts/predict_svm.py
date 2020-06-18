# import pandas as pd
import csv
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

train_X = []
test_X = []
train_Y = []
test_Y = []

######## Opening files and adding the data to array
# Open the files whose results are to be obtained
with open('training_data_15percent.csv', 'r') as f:
  reader = csv.reader(f)
  train_X = list(reader)

print(len(train_X))
print(len(train_X[0]))


with open('testing_data_15percent.csv', 'r') as f:
  reader = csv.reader(f)
  test_X = list(reader)

with open('training_data_labels_new.csv', 'r') as f:
  reader = csv.reader(f)
  temp_train_Y = list(reader)
  train_Y = [label for sublist in temp_train_Y for label in sublist]

with open('testing_data_labels_new.csv', 'r') as f:
  reader = csv.reader(f)
  temp_test_Y = list(reader)
  test_Y = [label for sublist in temp_test_Y for label in sublist]

# Train the model and fit data on this model
svm_model = SVC().fit(train_X, train_Y) 
svm_predictions = svm_model.predict(test_X) 

# Calculate results
print('Precision: ', precision_score(test_Y, svm_predictions, average='weighted'))

print('Recall: ', recall_score(test_Y, svm_predictions, average='weighted'))

print('F1 score: ', f1_score(test_Y, svm_predictions, average='weighted'))
