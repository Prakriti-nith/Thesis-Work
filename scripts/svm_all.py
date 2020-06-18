# import pandas as pd
import csv
import sys
from io import StringIO
from sklearn.svm import SVC 
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
with open('training_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  train_X_5grambinary = list(reader)

print(len(train_X_5grambinary))
print(len(train_X_5grambinary[0]))


with open('testing_data_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  test_X_5grambinary = list(reader)

with open('training_data_labels_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  temp_train_Y = list(reader)
  train_Y_5grambinary = [label for sublist in temp_train_Y for label in sublist]

with open('testing_data_labels_new_5gram.csv', 'r') as f:
  reader = csv.reader(f)
  temp_test_Y = list(reader)
  test_Y_5grambinary = [label for sublist in temp_test_Y for label in sublist]

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
svm_5grambinary = SVC(**kwargs, verbose=1).fit(train_X_5grambinary, train_Y_5grambinary) 
svm_predictions_5grambinary = svm_5grambinary.predict(test_X_5grambinary) 
sys.stdout = old_stdout
loss_history = mystdout.getvalue()
loss_list = []
for line in loss_history.split('\n'):
    if(len(line.split("loss: ")) == 1):
        continue
    loss_list.append(float(line.split("loss: ")[-1]))
print(loss_list)

# 93.83%
print('Test accuracy: ', accuracy_score(test_Y_5grambinary,svm_predictions_5grambinary))
# 88.09%
print('Precision: ', precision_score(test_Y_5grambinary, svm_predictions_5grambinary, average='weighted'))
# # 93.83%
print('Recall: ', recall_score(test_Y_5grambinary, svm_predictions_5grambinary, average='weighted'))
# # 90.87%
print('F1 score: ', f1_score(test_Y_5grambinary, svm_predictions_5grambinary, average='weighted'))

# loss_values_5grambinary = svm_5grambinary.loss_curve_
plt.plot(np.arange(len(loss_list)), loss_list)
plt.show()
# accuracy_vals_5grambinary = svm_5grambinary.validation_scores_
# plt.plot(accuracy_vals_5grambinary)
# plt.show()

# ############################################## 5gram multi-class
# train_X_5grammulti = []
# test_X_5grammulti = []
# train_Y_5grammulti = []
# test_Y_5grammulti = []
# with open('training_data_new_5gram.csv', 'r') as f:
#   reader = csv.reader(f)
#   train_X_5grammulti = list(reader)

# print(len(train_X_5grammulti))
# print(len(train_X_5grammulti[0]))


# with open('testing_data_new_5gram.csv', 'r') as f:
#   reader = csv.reader(f)
#   test_X_5grammulti = list(reader)

# with open('training_data_labels_new_5gram_multiclass.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_train_Y = list(reader)
#   train_Y_5grammulti = [label for sublist in temp_train_Y for label in sublist]

# with open('testing_data_labels_new_5gram_multiclass.csv', 'r') as f:
#   reader = csv.reader(f)
#   temp_test_Y = list(reader)
#   test_Y_5grammulti = [label for sublist in temp_test_Y for label in sublist]

# train_X_5grammulti = np.array(train_X_5grammulti, dtype=np.float32)
# test_X_5grammulti = np.array(test_X_5grammulti, dtype=np.float32)

# mlp_5grammulti = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150)
# mlp_5grammulti.fit(train_X_5grammulti, train_Y_5grammulti) 
# mlp_predictions_5grammulti = mlp_5grammulti.predict(test_X_5grammulti) 
 
# # 93.83%
# print('Test accuracy: ', accuracy_score(test_Y_5grammulti,mlp_predictions_5grammulti))
# # 88.09%
# print('Precision: ', precision_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))
# # # 93.83%
# print('Recall: ', recall_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))
# # # 90.87%
# print('F1 score: ', f1_score(test_Y_5grammulti, mlp_predictions_5grammulti, average='weighted'))

# loss_values_5grammulti = mlp_5grammulti.loss_curve_
# # print(loss_values_5grammulti)
# # plt.plot(loss_values_5grammulti)
# # plt.show()
# # accuracy_vals_5grambinary = mlp_5grambinary.validation_scores_
# # plt.plot(accuracy_vals_5grambinary)
# # plt.show()

# ############################################## 3gram binary
# train_X_3grambinary = []
# test_X_3grambinary = []
# train_Y_3grambinary = []
# test_Y_3grambinary = []
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

# mlp_3grambinary = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150)
# mlp_3grambinary.fit(train_X_3grambinary, train_Y_3grambinary) 
# mlp_predictions_3grambinary = mlp_3grambinary.predict(test_X_3grambinary) 

# # 93.83%
# print('Test accuracy: ', accuracy_score(test_Y_3grambinary,mlp_predictions_3grambinary))
# # 88.09%
# print('Precision: ', precision_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))
# # # 93.83%
# print('Recall: ', recall_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))
# # # 90.87%
# print('F1 score: ', f1_score(test_Y_3grambinary, mlp_predictions_3grambinary, average='weighted'))

# loss_values_3grambinary = mlp_3grambinary.loss_curve_
# # print(loss_values_3grambinary)
# # plt.plot(loss_values_3grambinary)
# # plt.show()
# # accuracy_vals_3grambinary = mlp_3grambinary.validation_scores_
# # plt.plot(accuracy_vals_3grambinary)
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

# mlp_3grammulti = MLPClassifier(solver='sgd', activation='logistic', learning_rate_init=.1, random_state=150)
# mlp_3grammulti.fit(train_X_3grammulti, train_Y_3grammulti) 
# mlp_predictions_3grammulti = mlp_3grammulti.predict(test_X_3grammulti) 
 
# # 93.83%
# print('Test accuracy: ', accuracy_score(test_Y_3grammulti,mlp_predictions_3grammulti))
# # 88.09%
# print('Precision: ', precision_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))
# # # 93.83%
# print('Recall: ', recall_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))
# # # 90.87%
# print('F1 score: ', f1_score(test_Y_3grammulti, mlp_predictions_3grammulti, average='weighted'))

# loss_values_3grammulti = mlp_3grammulti.loss_curve_
# # plt.plot(loss_values_3grammulti)
# # plt.show()
# # # accuracy_vals_3grammulti = mlp_3grammulti.validation_scores_
# # # plt.plot(accuracy_vals_3grammulti)
# # # plt.show()





# ############################################### plot loss function as 4 subplots
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
# ax1.plot(loss_values_3grambinary)
# ax1.set_title('3-gram binary')
# ax1.set_xlabel("Epochs", fontsize=12)
# ax1.set_ylabel("Loss", fontsize=12)
# ax2.plot(loss_values_3grammulti)
# ax2.set_title('3-gram multi-class')
# ax2.set_xlabel("Epochs", fontsize=12)
# ax2.set_ylabel("Loss", fontsize=12)
# ax3.plot(loss_values_5grambinary)
# ax3.set_title('5-gram binary')
# ax3.set_xlabel("Epochs", fontsize=12)
# ax3.set_ylabel("Loss", fontsize=12)
# ax4.plot(loss_values_5grammulti)
# ax4.set_title('5-gram multi-class')
# ax4.set_xlabel("Epochs", fontsize=12)
# ax4.set_ylabel("Loss", fontsize=12)
# # plt.xlabel("Epochs", fontsize=14)
# # plt.ylabel("Loss", fontsize=14)
# fig.tight_layout()
# # add a big axes, hide frame
# fig.add_subplot(111, frameon=False)
# # hide tick and tick label of the big axes
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
# plt.grid(False)
# # plt.xlabel("Epochs", fontsize=14, labelpad=10)
# # plt.ylabel("Loss", fontsize=14, labelpad=20)
# # fig.suptitle("", fontsize=14)
# plt.show()
# # 43.1%
# # print(accuracy)
