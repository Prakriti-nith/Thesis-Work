import matplotlib.pyplot as plt
import numpy as np

labels = ['Neural network', 'Decision tree', 'SVM']
mine = [94.39, 93.41, 89.61]
subba = [92.27, 90.23, 90.73]
bor = [92.41, 91.73, 91.51]
plt.plot(labels, mine, color='g', label='Proposed framework \n(3-gram terms | binary classification)')
plt.plot(labels, subba, color='orange', label='Subba B. [30] \n(Top 25% 3-gram terms)')
plt.plot(labels, bor, color='red', label='Borisaniya et al. [29]')
# plt.set_xticks([0, 1, 2])
# plt.set_xticklabels(labels)
plt.legend()
plt.xlabel('Classifiers', fontsize=14, labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=14, labelpad=10)
plt.show()