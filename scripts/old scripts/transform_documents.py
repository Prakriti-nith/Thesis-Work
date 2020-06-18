from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
from sklearn.pipeline import Pipeline
import numpy as np
import random
import csv

vocabulary = []

with open('../features_tfidf.txt', "r") as f:
	vocabulary = f.read().split(',')

# print(vocabulary)
# print(len(vocabulary))

# Removing duplicates
# Total features now: 1925
vocabulary = list(dict.fromkeys(vocabulary))
# print(len(vocabulary)) 

# shuffle them
random.shuffle(vocabulary)
# print(vocabulary)

filenames_normal = []
filenames_attack = []

# Training normal
for x in range(1,834):
    if x<10:
        filenames_normal.append("../Training_Data_Master/UTD-000" + str(x) + ".txt")
    elif x<100:
        filenames_normal.append("../Training_Data_Master/UTD-00" + str(x) + ".txt")
    else:
        filenames_normal.append("../Training_Data_Master/UTD-0" + str(x) + ".txt")

# Validation normal
for x in range(1,4373):
    if x<10:
        filenames_normal.append("../Validation_Data_Master/UVD-000" + str(x) + ".txt")
    elif x<100:
        filenames_normal.append("../Validation_Data_Master/UVD-00" + str(x) + ".txt")
    elif x<1000:
        filenames_normal.append("../Validation_Data_Master/UVD-0" + str(x) + ".txt")
    else:
        filenames_normal.append("../Validation_Data_Master/UVD-" + str(x) + ".txt")  

# print(len(filenames_normal)) -> 5205
# cnt = 0

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Adduser_' + str(i) + '/*.txt'))
# print(filenames_attack)
########## Total Adduser documents -> 91
# for i in range(0,10):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt) -> 91 adduser documents

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Hydra_FTP_' + str(i) + '/*.txt'))
########## Total Hydra FTP documents -> 162
# for i in range(10,20):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt)

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Hydra_SSH_' + str(i) + '/*.txt'))
########## Total Hydra SSH documents -> 176
# for i in range(20,30):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt)

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Java_Meterpreter_' + str(i) + '/*.txt'))
########## Total Hydra SSH documents -> 124
# for i in range(30,40):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt)

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Meterpreter_' + str(i) + '/*.txt'))
########## Total Hydra SSH documents -> 75
# for i in range(40,50):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt)

for i in range(1,11):
	filenames_attack.append(glob('../Attack_Data_Master/Web_Shell_' + str(i) + '/*.txt'))
########## Total Hydra SSH documents -> 118
# for i in range(50,60):
# 	cnt = cnt + len(filenames_attack[i])
# print(cnt)
# print(len(filenames_attack)) -> 60

corpus = []

for fname in filenames_normal:
    with open(fname, 'r') as file:
    	corpus.append(file.read().replace('\n', ''))

for i in range(len(filenames_attack)):
	for fname in filenames_attack[i]:
	    with open(fname, 'r') as file:
	    	corpus.append(file.read().replace('\n', ''))

# print(corpus)
print(len(corpus))


ml_data = []
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary, ngram_range=(3,3))), ('tfid', TfidfTransformer())]).fit(corpus)
ml_data = pipe['count'].transform(corpus).toarray().tolist()
# print(ml_data)
# print(type(ml_data))
# print(len(ml_data))
# print(len(ml_data[2]))

''' Classes
0 -> normal
1 -> adduser
2 -> hydraftp
3 -> hydrassh
4 -> java_meterpreter
5 -> meterpreter
6 -> web shell '''

count=0
for row in ml_data:
	count = count+1
	if count<=5205:
		row.append(0)
	elif count<=5296:
		row.append(1)
	elif count<=5458:
		row.append(2)
	elif count<=5634:
		row.append(3)
	elif count<=5758:
		row.append(4)
	elif count<=5833:
		row.append(5)
	else:
		row.append(6)



# Remove rows with all zeros except last element as it is the class
# print(ml_data)
# for row_features in ml_data:
# 	cnt = 0
# 	for i in range(0,1925):
# 		if row_features[i] != 0:
# 			break
# 		else:
# 			cnt += 1
# 	if cnt == 1925:
# 		ml_data.remove(row_features)

# print(len(ml_data))


ml_data_train = []
ml_data_test = []
cnt = 0
# Split data into training and testing
for row in ml_data:
	cnt = cnt+1
	if cnt<=833:
		ml_data_train.append(row)
	elif cnt>5205 and cnt<=5255:
		ml_data_train.append(row)
	elif cnt>5296 and cnt<=5385:
		ml_data_train.append(row)
	elif cnt>5458 and cnt<=5555:
		ml_data_train.append(row)
	elif cnt>5634 and cnt<=5702:
		ml_data_train.append(row)
	elif cnt>5758 and cnt<=5799:
		ml_data_train.append(row)
	elif cnt>5833 and cnt<=5898:
		ml_data_train.append(row)

cnt=0
for row in ml_data:
	cnt = cnt+1
	if cnt>833 and cnt<=5205:
		ml_data_test.append(row)
	elif cnt>5255 and cnt<=5296:
		ml_data_test.append(row)
	elif cnt>5385 and cnt<=5458:
		ml_data_test.append(row)
	elif cnt>5555 and cnt<=5634:
		ml_data_test.append(row)
	elif cnt>5702 and cnt<=5758:
		ml_data_test.append(row)
	elif cnt>5799 and cnt<=5833:
		ml_data_test.append(row)
	elif cnt>5898:
		ml_data_test.append(row)


# Remove rows with all zeros except last element as it is the class
print("ml data length: " + str(len(ml_data)))
print("ml training data length: " + str(len(ml_data_train)))
print("ml testing data length: " + str(len(ml_data_test)))
for row_features in ml_data_train:
	cnt = 0
	for i in range(0,1925):
		if row_features[i] != 0:
			break
		else:
			cnt += 1
	if cnt == 1925:
		ml_data_train.remove(row_features)

for row_features in ml_data_test:
	cnt = 0
	for i in range(0,1925):
		if row_features[i] != 0:
			break
		else:
			cnt += 1
	if cnt == 1925:
		ml_data_test.remove(row_features)

print("ml training data length after removing zeros: " + str(len(ml_data_train)))
print("ml testing data length after removing zeros: " + str(len(ml_data_test)))


# Move class labels to different list
ml_training_labels = []
ml_testing_labels = []
for row_features in ml_data_train:
	temp = []
	temp.append(row_features[-1])
	ml_training_labels.append(temp)
	del row_features[-1]
for row_features in ml_data_test:
	temp = []
	temp.append(row_features[-1])
	ml_testing_labels.append(temp)
	del row_features[-1]

print("ml training data final length: " + str(len(ml_data_train)))
print("ml testing data final length: " + str(len(ml_data_test)))
print("ml training labels data final length: " + str(len(ml_data_train)))
print("ml testing data final length: " + str(len(ml_data_test)))


# Place the data in files
# with open("training_data.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_data_train)

# with open("testing_data.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_data_test)

# with open("training_data_labels.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_training_labels)

# with open("testing_data_labels.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerows(ml_testing_labels)



# New File(s) work
''' 
1. Separate the data for training and testing in two different files
2. Apply different ML algorithms on them:
	a. Neural net - hyper parameters activation function
	b. SVM - kernel variation
	c. Decision tree
'''
