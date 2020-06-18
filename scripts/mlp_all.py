# import pandas as pd
import csv
from sklearn import model_selection, linear_model
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

########################################## 5gram binary
train_X_5grambinary = []
test_X_5grambinary = []
train_Y_5grambinary = []
test_Y_5grambinary = []

######## Opening files and adding the data to array
# Open the files whose results are to be obtained
with open('training_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  train_X_5grambinary = list(reader)

print(len(train_X_5grambinary))
print(len(train_X_5grambinary[0]))


with open('testing_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  test_X_5grambinary = list(reader)

with open('training_data_labels_new.csv', 'r') as f:
  reader = csv.reader(f)
  temp_train_Y = list(reader)
  train_Y_5grambinary = [label for sublist in temp_train_Y for label in sublist]

with open('testing_data_labels_new.csv', 'r') as f:
  reader = csv.reader(f)
  temp_test_Y = list(reader)
  test_Y_5grambinary = [label for sublist in temp_test_Y for label in sublist]

train_X_5grambinary = np.array(train_X_5grambinary, dtype=np.float32)
test_X_5grambinary = np.array(test_X_5grambinary, dtype=np.float32)

###### splitting training and validation set
X_tr_5grambinary,X_val_5grambinary,y_tr_5grambinary,y_val_5grambinary=model_selection.train_test_split(train_X_5grambinary,train_Y_5grambinary,test_size=0.2)

###### Train the model and fit data on this model
mlp_5grambinary = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150, max_iter=28)
mlp_5grambinary.fit(X_tr_5grambinary, y_tr_5grambinary) 
mlp_predictions_5grambinary = mlp_5grambinary.predict(test_X_5grambinary) 

###### Calculate results
print('Precision: ', precision_score(test_Y_5grambinary, mlp_predictions_5grambinary, average='weighted'))

print('Recall: ', recall_score(test_Y_5grambinary, mlp_predictions_5grambinary, average='weighted'))

print('F1 score: ', f1_score(test_Y_5grambinary, mlp_predictions_5grambinary, average='weighted'))

loss_values_5grambinary = mlp_5grambinary.loss_curve_
# plt.plot(loss_values_5grambinary)
# plt.show()

mlp_5grambinary.fit(X_val_5grambinary, y_val_5grambinary)
loss_values_5binary_val = mlp_5grambinary.loss_curve_
plt.plot(loss_values_5grambinary)
plt.plot(loss_values_5binary_val)
# plt.show()

############################################## 5gram multi-class
train_X_5grammulti = []
test_X_5grammulti = []
train_Y_5grammulti = []
test_Y_5grammulti = []

######## Opening files and adding the data to array
# Open the files whose results are to be obtained
with open('training_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  train_X_5grammulti = list(reader)

print(len(train_X_5grammulti))
print(len(train_X_5grammulti[0]))


with open('testing_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  test_X_5grammulti = list(reader)

with open('training_data_labels_new_multiclass.csv', 'r') as f:
  reader = csv.reader(f)
  temp_train_Y = list(reader)
  train_Y_5grammulti = [label for sublist in temp_train_Y for label in sublist]

with open('testing_data_labels_new_multiclass.csv', 'r') as f:
  reader = csv.reader(f)
  temp_test_Y = list(reader)
  test_Y_5grammulti = [label for sublist in temp_test_Y for label in sublist]

train_X_5grammulti = np.array(train_X_5grammulti, dtype=np.float32)
test_X_5grammulti = np.array(test_X_5grammulti, dtype=np.float32)

###### splitting training and validation set
X_tr_5grammulti,X_val_5grammulti,y_tr_5grammulti,y_val_5grammulti=model_selection.train_test_split(train_X_5grammulti,train_Y_5grammulti,test_size=0.2)

###### Train the model and fit data on this model
mlp_5grammulti = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150, max_iter=45)
mlp_5grammulti.fit(X_tr_5grammulti, y_tr_5grammulti) 
mlp_predictions_5grammulti = mlp_5grammulti.predict(test_X_5grammulti) 

###### Calculate results
print('Precision: ', precision_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))

print('Recall: ', recall_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))

print('F1 score: ', f1_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))

loss_values_5grammulti = mlp_5grammulti.loss_curve_
# print(loss_values_5grammulti)
# plt.plot(loss_values_5grammulti)
# plt.show()

mlp_5grammulti.fit(X_val_5grammulti, y_val_5grammulti)
loss_values_5grammulti_val = mlp_5grammulti.loss_curve_
plt.plot(loss_values_5grammulti)
plt.plot(loss_values_5grammulti_val)
# plt.show()

# ############################################## 3gram binary
# train_X_3grambinary = []
# test_X_3grambinary = []
# train_Y_3grambinary = []
# test_Y_3grambinary = []

# ######## Opening files and adding the data to array
# # Open the files whose results are to be obtained
# with open('training_data_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   train_X_3grambinary = list(reader)

# print(len(train_X_3grambinary))
# print(len(train_X_3grambinary[0]))


# with open('testing_data_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   test_X_3grambinary = list(reader)

# with open('training_data_labels_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_train_Y = list(reader)
#   train_Y_3grambinary = [label for sublist in temp_train_Y for label in sublist]

# with open('testing_data_labels_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_test_Y = list(reader)
#   test_Y_3grambinary = [label for sublist in temp_test_Y for label in sublist]

# train_X_3grambinary = np.array(train_X_3grambinary, dtype=np.float32)
# test_X_3grambinary = np.array(test_X_3grambinary, dtype=np.float32)

# ###### splitting training and validation set
# X_tr_3grambinary,X_val_3grambinary,y_tr_3grambinary,y_val_3grambinary=model_selection.train_test_split(train_X_3grambinary,train_Y_3grambinary,test_size=0.2)

# ###### Train the model and fit data on this model
# mlp_3grambinary = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150, max_iter=35)
# mlp_3grambinary.fit(X_tr_3grambinary, y_tr_3grambinary) 
# mlp_predictions_3grambinary = mlp_3grambinary.predict(test_X_3grambinary) 

# ###### Calculate results
# print('Precision: ', precision_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))

# print('Recall: ', recall_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))

# print('F1 score: ', f1_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))

# loss_values_3grambinary = mlp_3grambinary.loss_curve_
# # print(loss_values_3grambinary)
# # plt.plot(loss_values_3grambinary)
# # plt.show()

# mlp_3grambinary.fit(X_val_3grambinary, y_val_3grambinary)
# loss_values_3grambinary_val = mlp_3grambinary.loss_curve_
# plt.plot(loss_values_3grambinary)
# plt.plot(loss_values_3grambinary_val)
# # plt.show()


# # ############################################## 3gram multiclass
# train_X_3grammulti = []
# test_X_3grammulti = []
# train_Y_3grammulti = []
# test_Y_3grammulti = []
# with open('training_data_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   train_X_3grammulti = list(reader)

# print(len(train_X_3grammulti))
# print(len(train_X_3grammulti[0]))


# with open('testing_data_new.csv', 'r') as f:
#   reader = csv.reader(f)
#   test_X_3grammulti = list(reader)

# with open('training_data_labels_new_multiclass.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_train_Y = list(reader)
#   train_Y_3grammulti = [label for sublist in temp_train_Y for label in sublist]

# with open('testing_data_labels_new_multiclass.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_test_Y = list(reader)
#   test_Y_3grammulti = [label for sublist in temp_test_Y for label in sublist]

# train_X_3grammulti = np.array(train_X_3grammulti, dtype=np.float32)
# test_X_3grammulti = np.array(test_X_3grammulti, dtype=np.float32)

# ###### splitting training and validation set
# X_tr_3grammulti,X_val_3grammulti,y_tr_3grammulti,y_val_3grammulti=model_selection.train_test_split(train_X_3grammulti,train_Y_3grammulti,test_size=0.2)

# ###### Train the model and fit data on this model
# mlp_3grammulti = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150, max_iter=41)
# mlp_3grammulti.fit(X_tr_3grammulti, y_tr_3grammulti) 
# mlp_predictions_3grammulti = mlp_3grammulti.predict(test_X_3grammulti) 

# ###### Calculate results
# print('Precision: ', precision_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))

# print('Recall: ', recall_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))

# print('F1 score: ', f1_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))

# loss_values_3grammulti = mlp_3grammulti.loss_curve_
# # plt.plot(loss_values_3grammulti)
# # plt.show()

# mlp_3grammulti.fit(X_val_3grammulti, y_val_3grammulti)
# loss_values_3grammulti_val = mlp_3grammulti.loss_curve_
# plt.plot(loss_values_3grammulti)
# plt.plot(loss_values_3grammulti_val)
# # plt.show()




############################################### plot loss function as 4 subplots
fig, (ax3, ax4) = plt.subplots(1,2)
ax3.plot(loss_values_5grambinary, label='training loss')
ax3.plot(loss_values_5binary_val, label='validation loss')
ax3.set_title('5-gram binary')
ax3.set_xlabel("Epochs", fontsize=12)
ax3.set_ylabel("Loss", fontsize=12)
ax3.legend()
ax4.plot(loss_values_5grammulti, label='training loss')
ax4.plot(loss_values_5grammulti_val, label='validation loss')
ax4.set_title('5-gram multi-class')
ax4.set_xlabel("Epochs", fontsize=12)
ax4.set_ylabel("Loss", fontsize=12)
ax4.legend()
fig.tight_layout()
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.show()

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# ax1.plot(loss_values_3grambinary, label='training loss')
# ax1.plot(loss_values_3grambinary_val, label='validation loss')
# ax1.set_title('3-gram binary')
# ax1.set_xlabel("Epochs", fontsize=12)
# ax1.set_ylabel("Loss", fontsize=12)
# ax1.legend()
# ax2.plot(loss_values_3grammulti, label='training loss')
# ax2.plot(loss_values_3grammulti_val, label='validation loss')
# ax2.set_title('3-gram multi-class')
# ax2.set_xlabel("Epochs", fontsize=12)
# ax2.set_ylabel("Loss", fontsize=12)
# ax2.legend()
# ax3.plot(loss_values_5grambinary, label='training loss')
# ax3.plot(loss_values_5binary_val, label='validation loss')
# ax3.set_title('5-gram binary')
# ax3.set_xlabel("Epochs", fontsize=12)
# ax3.set_ylabel("Loss", fontsize=12)
# ax3.legend()
# ax4.plot(loss_values_5grammulti, label='training loss')
# ax4.plot(loss_values_5grammulti_val, label='validation loss')
# ax4.set_title('5-gram multi-class')
# ax4.set_xlabel("Epochs", fontsize=12)
# ax4.set_ylabel("Loss", fontsize=12)
# ax4.legend()
# fig.tight_layout()
# fig.add_subplot(111, frameon=False)
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.grid(False)
# plt.show()
