import numpy as np
import matplotlib.pyplot as plt

############## 3grams

labels = ['Neural network', 'SVM', 'Decision tree']
dr_binary_3gram = [96.08, 89.57, 93.49]
acc_binary_3gram = [96.00, 90.52, 93.23]
fmeasure_binary_3gram = [96.03, 86.77, 93.31]
dr_multi_3gram = [92.00, 88.19, 89.52]
acc_multi_3gram = [90.36, 83.03, 88.35]
fmeasure_multi_3gram = [90.63, 83.93, 88.84]

x = np.arange(len(labels))  # the label locations
width = 0.14  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2 - 2*width, acc_binary_3gram, width, label='Accuracy | binary classification')
rects2 = ax.bar(x - width/2 - width, dr_binary_3gram, width, label='Detection Rate | binary classification')
rects3 = ax.bar(x - width/2, fmeasure_binary_3gram, width, label='F-measure | binary classification')
rects4 = ax.bar(x + width/2 , acc_multi_3gram, width, label='Accuracy | multi-class classification')
rects5 = ax.bar(x + width/2 + width, dr_multi_3gram, width, label='Detection Rate | multi-class classification')
rects6 = ax.bar(x + width/2 + 2*width, fmeasure_multi_3gram, width, label='F-measure | multi-class classification')

fig.tight_layout()

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim([0,120])
# ax.set_title('Accuracy and F1-measure of 3-gram model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()

plt.xlabel('Classifiers', fontsize=14, labelpad=10)
plt.ylabel('Acc/DR/F-measure (%)', fontsize=14)
plt.show()

################ 5grams
labels = ['Neural network', 'SVM', 'Decision tree']
acc_binary_3gram = [95.69, 89.61, 93.41]
dr_binary_3gram = [95.80, 88.19, 93.71]
fmeasure_binary_3gram = [95.66, 84.19, 93.42]
acc_multi_3gram = [90.74, 80.61, 85.49]
dr_multi_3gram = [91.94, 86.65, 88.03]
fmeasure_multi_3gram = [90.74, 80.87, 86.60]

x = np.arange(len(labels))  # the label locations
width = 0.14  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2 - 2*width, acc_binary_3gram, width, label='Accuracy | binary classification')
rects2 = ax.bar(x - width/2 - width, dr_binary_3gram, width, label='Detection Rate | binary classification')
rects3 = ax.bar(x - width/2, fmeasure_binary_3gram, width, label='F-measure | binary classification')
rects4 = ax.bar(x + width/2 , acc_multi_3gram, width, label='Accuracy | multi-class classification')
rects5 = ax.bar(x + width/2 + width, dr_multi_3gram, width, label='Detection Rate | multi-class classification')
rects6 = ax.bar(x + width/2 + 2*width, fmeasure_multi_3gram, width, label='F-measure | multi-class classification')

fig.tight_layout()

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylim([0,120])
# ax.set_title('Accuracy and F1-measure of 3-gram model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

fig.tight_layout()

plt.xlabel('Classifiers', fontsize=14, labelpad=10)
plt.ylabel('Acc/DR/F-measure (%)', fontsize=14)
plt.show()
