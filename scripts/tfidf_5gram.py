from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from glob import glob
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
import csv

filenames = []
data = []

### normal files
print("training normal files reading...")
for x in range(1,601):
    if x<10:
        filenames.append("../Training_Data_Master/UTD-000" + str(x) + ".txt")
    elif x<100:
        filenames.append("../Training_Data_Master/UTD-00" + str(x) + ".txt")
    else:
        filenames.append("../Training_Data_Master/UTD-0" + str(x) + ".txt")

# print("validation normal files reading...")
# for x in range(1,501):
#     if x<10:
#         filenames.append("../Validation_Data_Master/UVD-000" + str(x) + ".txt")
#     elif x<100:
#         filenames.append("../Validation_Data_Master/UVD-00" + str(x) + ".txt")
#     elif x<1000:
#         filenames.append("../Validation_Data_Master/UVD-0" + str(x) + ".txt")
#     else:
#         filenames.append("../Validation_Data_Master/UVD-" + str(x) + ".txt")  

print("reading normal data...")
for fname in filenames:
    if fname == '../Training_Data_Master/outfile.txt':
        # don't want to copy the output into the output
        continue
    with open(fname, 'r') as file:
    	data.append(file.read().replace('\n', ''))

##### adduser
filenames.clear()

print("adduser files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Adduser_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

##### hydra-ftp
filenames.clear()

print("hydra-ftp files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

#### hydra-ssh
filenames.clear()

print("hydra-ssh files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

#### java meterpreter
filenames.clear()

print("java meterpreter files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

#### meterpreter
filenames.clear()

print("meterpreter files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Meterpreter_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

#### web-shell
filenames.clear()

print("web-shell files reading...")
for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Web_Shell_' + str(i) + '/*.txt'))

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

print("done reading data")

############## Fit data
print("Fitting data...")
tfidf_vectorizer = TfidfVectorizer(norm=None, ngram_range=(7,7))
new_term_freq_matrix = tfidf_vectorizer.fit_transform(data)

print("Fitting data done. Shape of the matrix generated is: ")
print(new_term_freq_matrix.shape)

# ########## Dimensionality reduction
print("Performing dimensionality reduction...")
svd = TruncatedSVD(n_components=2995, n_iter=5, random_state=42)
new_term_freq_matrix = svd.fit_transform(new_term_freq_matrix)

print("Dimensionality reduction using svd done. Shape of the new matrix generated is: ")
print(new_term_freq_matrix.shape)
print(type(new_term_freq_matrix))

# #### numpy.ndarray to list
# ml_data = new_term_freq_matrix.tolist()
# # print(ml_data)
# print(len(ml_data))
# print(len(ml_data[0]))

# #### Append classes
# ''' Classes
# 0 -> normal
# 1 -> adduser
# 2 -> hydraftp
# 3 -> hydrassh
# 4 -> java_meterpreter
# 5 -> meterpreter
# 6 -> web shell '''

# print("Appending classes...")

# count=0
# for row in ml_data:
# 	count = count+1
# 	if count<=5205:
# 		row.append(0)
# 	# else:
# 	# 	row.append(1)
# 	elif count<=5296:
# 		row.append(1)
# 	elif count<=5458:
# 		row.append(2)
# 	elif count<=5634:
# 		row.append(3)
# 	elif count<=5758:
# 		row.append(4)
# 	elif count<=5833:
# 		row.append(5)
# 	else:
# 		row.append(6)

# #### Split into train and test data
# print("Splitting data into train and test...")
# ml_data_train = []
# ml_data_test = []
# cnt = 0

# for row in ml_data:
# 	cnt = cnt+1
# 	if cnt<=3643:
# 		ml_data_train.append(row)
# 	elif cnt>5205 and cnt<=5262:
# 		ml_data_train.append(row)
# 	elif cnt>5296 and cnt<=5405:
# 		ml_data_train.append(row)
# 	elif cnt>5458 and cnt<=5578:
# 		ml_data_train.append(row)
# 	elif cnt>5634 and cnt<=5719:
# 		ml_data_train.append(row)
# 	elif cnt>5758 and cnt<=5805:
# 		ml_data_train.append(row)
# 	elif cnt>5833 and cnt<=5910:
# 		ml_data_train.append(row)

# cnt=0
# for row in ml_data:
# 	cnt = cnt+1
# 	if cnt>3643 and cnt<=5205:
# 		ml_data_test.append(row)
# 	elif cnt>5262 and cnt<=5296:
# 		ml_data_test.append(row)
# 	elif cnt>5405 and cnt<=5458:
# 		ml_data_test.append(row)
# 	elif cnt>5578 and cnt<=5634:
# 		ml_data_test.append(row)
# 	elif cnt>5719 and cnt<=5758:
# 		ml_data_test.append(row)
# 	elif cnt>5805 and cnt<=5833:
# 		ml_data_test.append(row)
# 	elif cnt>5910:
# 		ml_data_test.append(row)

# ##### Move class labels to different list
# print("Moving class labels to different list...")
# ml_training_labels = []
# ml_testing_labels = []
# for row_features in ml_data_train:
# 	temp = []
# 	temp.append(row_features[-1])
# 	ml_training_labels.append(temp)
# 	del row_features[-1]
# for row_features in ml_data_test:
# 	temp = []
# 	temp.append(row_features[-1])
# 	ml_testing_labels.append(temp)
# 	del row_features[-1]

# print("ml training data final length: " + str(len(ml_data_train)))
# print("ml testing data final length: " + str(len(ml_data_test)))
# print("ml training labels data final length: " + str(len(ml_training_labels)))
# print("ml testing labels data final length: " + str(len(ml_testing_labels)))

# ## Place the data in files
# print("Writing training data to csv file...")
# with open("training_data_new_5gram.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_data_train)

# print("Writing testing data to csv file...")
# with open("testing_data_new_5gram.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_data_test)

# print("Writing training data labels to csv file...")
# with open("training_data_labels_new_5gram_multiclass.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_training_labels)

# print("Writing testing data labels to csv file...")
# with open("testing_data_labels_new_5gram_multiclass.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_testing_labels)



# New File(s) work
''' 
1. Apply different ML algorithms on them:
	a. Neural net - hyper parameters activation function
	b. SVM - kernel variation
	c. Decision tree
'''
