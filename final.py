
# coding: utf-8

# # V.1 Exploring the green reds
# ## a) function that plot scatterplot matrix of red wine data

# In[37]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
    df = pd.read_csv(wine_data, sep=';')
    numvars, numdata = df.shape
    print('numvars:{} numdata:{}'.format(numvars, numdata))
    fig, axes = plt.subplots(numdata, numdata, figsize=(50,50))
    for i in range(numdata):
        for j in range(numdata):
            if i == j:
                axes[i, j].text(0.5, 0.5, df.columns[i], ha='center')
            else:
                color = ['green' if k == good_threshold else 'red' for k in df['quality']]
                size = [10 if k == good_threshold else 10 if k == bad_threshold else 0 for k in df['quality']]
                axes[i,j].scatter(df[df.columns[i]], df[df.columns[j]], c = color, s = size)
    if save_plot==True:
        plt.savefig('./winedata.png')
    plt.tight_layout()
    fig

plot_scatter_matrix('winequality-red.csv', 8, 3, save_plot=False)


# ## b) the most useful factor for distinguishing high vs. low quality wine and why?
# 
