import matplotlib.pyplot as plt
import numpy as np

labels = ['Neural network', 'Decision tree', 'SVM']
mine = [94.53, 93.71, 88.19]
subba = [90.15, 93.19, 88.10]
bor = [91.13, 93.11, 89.83]
plt.plot(labels, mine, color='g', label='Proposed framework \n(3-gram terms | binary classification)')
plt.plot(labels, subba, color='orange', label='Subba B. [30] \n(Top 25% 3-gram terms)')
plt.plot(labels, bor, color='red', label='Borisaniya et al. [29]')
# plt.set_xticks([0, 1, 2])
# plt.set_xticklabels(labels)
plt.legend()
plt.xlabel('Classifiers', fontsize=14, labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
plt.show()