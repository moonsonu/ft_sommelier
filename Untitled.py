
# coding: utf-8

# # V.1:Exploring the green reds
# a) function that plot a scatterplot matric of red wine data

# In[3]:



import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
df = pd.read_csv('winequality-red.csv', sep=';')
df
    


# In[25]:


import itertools
import matplotlib.pyplot as plt
import numpy as np
def plot_scatter_matrix(data, names, **kwargs):
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(8,8))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            axes[x,y].plot(data[x], data[y], **kwargs)
    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    return fig
df = pd.read_csv('winequality-red.csv', sep=';')
np.random.seed(1599)
numvars, numdata = 4, 10 
data = np.random.random((numvars, numdata))
fig = plot_scatter_matrix(data, ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar'], linestyle='none', marker='o', color='black', mfc='none')
plt.show()


# In[ ]:


def plot_scatter_matrix(wine_data, good_threshold, bad_threshhold, save_plot=False):


# In[26]:


get_ipython().run_line_magic('pwd', '')

