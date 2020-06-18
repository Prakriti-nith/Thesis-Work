from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from glob import glob
import numpy as np


tfidf_vectorizer = TfidfVectorizer(norm=None, ngram_range=(3,3))
filenames = []

for i in range(1,11):
	filenames.append(glob('../Attack_Data_Master/Adduser_' + str(i) + '/*.txt'))

print(filenames)

data = []

for i in range(len(filenames)):
	for fname in filenames[i]:
	    with open(fname, 'r') as file:
	    	data.append(file.read().replace('\n', ''))

new_term_freq_matrix = tfidf_vectorizer.fit_transform(data)
# print(tfidf_vectorizer.vocabulary_)

# print(tfidf_vectorizer.idf_)
print(len(tfidf_vectorizer.idf_))

##### Picking top 15% values as total is 562
# indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
# features = tfidf_vectorizer.get_feature_names()
# top_n = 90
# top_features = [features[i] for i in indices[:top_n]]
# print(top_features)

# with open('../features_tfidf.txt', 'a') as outfile:
#     for feature in top_features:
#         outfile.write("%s," % (feature))
