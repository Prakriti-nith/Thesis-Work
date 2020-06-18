import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = ('Proposed \n(3-gram terms)', 'Subba B. et al. [30] \n(Top 25% 3-gram terms)', 'Borisaniya et al. [29]')
y_pos = np.arange(len(people))
no_terms = [550, 1454, 7920]
width = 0.5

ax.barh(y_pos, no_terms, width, align='center', edgecolor='black', color='#C0C0C0', zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Feature count', fontsize=14, labelpad=5)
ax.set_ylabel('HIDS frameworks', fontsize=14)
ax.set_xlim([0,9000])
ax.grid(zorder=0,linestyle='--', linewidth=0.5, color='#D4D4D4')

# for i, v in enumerate(no_terms):
#     ax.text(v + 10, i, str(v))

plt.show()
