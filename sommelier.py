
# coding: utf-8

# # V.1 Exploring the green reds
# ## a) function that plot scatterplot matrix of red wine data

# In[4]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot):
    df = pd.read_csv(wine_data, sep=';')
    numvars, numdata = df.shape
    fig, axes = plt.subplots(numdata, numdata, figsize=(50,50))
    for i in range(numdata):
        for j in range(numdata):
            if i == j:
                axes[i, j].text(0.5, 0.5, df.columns[i], ha='center')
            else:
                color = ['blue' if k >= good_threshold else 'red' for k in df['quality']]
                size = [10 if k >= good_threshold else 10 if k <= bad_threshold else 0 for k in df['quality']]
                axes[i,j].scatter(df[df.columns[i]], df[df.columns[j]], c = color, s = size)
    if save_plot==True:
        plt.savefig('./winedata.png')
    plt.tight_layout()
    fig

plot_scatter_matrix('winequality-red.csv', 8, 3, save_plot=False)


# ## b) the most useful factor for distinguishing high vs. low quality wine and why?
# ### According to the graph of factors and quality, alcohol is the most useful factor for distinguishing high and low quality wine. The distribution of values in alcohol in high score which is 8 or higher and in low score which is 3 or lower is quite distinct from each other. By the alcohol value of 11, above it is consider as high quality and below it is consider as low quality.
# 

# # V.2 Learning to perceptron
# ## a) & b) implement a perceptron

# In[70]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

class perceptron():
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=10):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = []
        for i in range(1 + X.shape[1]):
            self.w_.append(random.random())
        print(self.w_)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (int(target) - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            #print(self.w_)
        return self
    def net_input(self, X):
        activation = self.w_[0]
        for i in range(len(X)):
            activation += self.w_[i + 1] * X[i]
        return activation
    def predict(self, X):
        if self.net_input(X) > self.thresholds:
            return 1
        else:
            return -1
df = pd.read_csv('winequality-red.csv', sep=';')
df_filtered = df[(df.quality >= 8) | (df.quality <= 3)]
df_data = df_filtered[['alcohol', 'pH', 'quality']]
y = ['1' if k >= 8 else '-1' for k in df_data['quality']]
X = df_data[['alcohol', 'pH']].values
p = perceptron(eta=0.1)
p.fit(X, y)
print(p.errors_)


# In[ ]:




