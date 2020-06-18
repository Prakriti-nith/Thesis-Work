from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

tfidf_vectorizer = TfidfVectorizer(norm=None, ngram_range=(3,3))
filenames = []

for x in range(1,834):
    if x<10:
        filenames.append("../Training_Data_Master/UTD-000" + str(x) + ".txt")
    elif x<100:
        filenames.append("../Training_Data_Master/UTD-00" + str(x) + ".txt")
    else:
        filenames.append("../Training_Data_Master/UTD-0" + str(x) + ".txt")

for x in range(1,4373):
    if x<10:
        filenames.append("../Validation_Data_Master/UVD-000" + str(x) + ".txt")
    elif x<100:
        filenames.append("../Validation_Data_Master/UVD-00" + str(x) + ".txt")
    elif x<1000:
        filenames.append("../Validation_Data_Master/UVD-0" + str(x) + ".txt")
    else:
        filenames.append("../Validation_Data_Master/UVD-" + str(x) + ".txt")        

data = []

for fname in filenames:
    if fname == '../Training_Data_Master/outfile.txt':
        # don't want to copy the output into the output
        continue
    with open(fname, 'r') as file:
    	data.append(file.read().replace('\n', ''))

# print(data)
 
new_term_freq_matrix = tfidf_vectorizer.fit_transform(data)
# print(tfidf_vectorizer.vocabulary_)
print(len(tfidf_vectorizer.vocabulary_))

# print(tfidf_vectorizer.idf_)
# print(len(tfidf_vectorizer.idf_))

##### Picking top 15% values as total is 7464
# indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
# features = tfidf_vectorizer.get_feature_names()
# top_n = 1119
# top_features = [features[i] for i in indices[:top_n]]
# print(top_features)

# with open('../features_tfidf.txt', 'a') as outfile:
#     for feature in top_features:
#         outfile.write("%s," % (feature))
