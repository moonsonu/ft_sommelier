
# coding: utf-8

# # V.1 Exploring the green reds
# ### a) function that plot scatterplot matrix of red wine data

# In[79]:


import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import random


# In[119]:


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
# ### a) & b) implement a perceptron

# In[193]:


class perceptron():
    def __init__(self, thresholds=0.0, eta=0.01, n_iter=14000):
        self.thresholds = thresholds
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        self.w_ = []
        result_ = []
        num_epoch = 0
        for i in range(1 + X.shape[1]):
            self.w_.append(random.random())
        self.errors_ = []
        if (self.n_iter != 0):
            for _ in range(self.n_iter):
                errors = 0
                result = 0
                num_epoch += 1
                for xi, target in zip(X, y):
                    update = self.eta * (int(target) - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                    result = (num_epoch, errors, self.w_[1:], self.w_[0])
                self.errors_.append(errors)
                result_.append(result)
                
        else:
            while (1):
                errors = 0
                result = 0
                num_poch += 1
                for xi, target in zip(X, y):
                    update = self.eta * (int(target) - self.predict(xi))
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    errors += int(update != 0.0)
                    result = (num_epoch, errors, self.w_[1:], self.w_[0])
                self.errors_.append(errors)
                result_.append(result)
                if (errors == 0):
                    break
                
                
                #print(self.w_)
        
        return result_
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
            
p = perceptron(eta=0.01)


# ### c) function that will take the output of perceptron training function and generate two plots in one figure

# In[217]:


def plot_performance(performance, wine_data, good_thresh, bad_thresh, epoch=-1, save_plot=False):
    df = pd.read_csv(wine_data, sep=';')
    df_filtered = df[(df.quality >= good_thresh) | (df.quality <= bad_thresh)]
    df_data = df_filtered[['alcohol', 'pH', 'quality']]
    y = ['1' if k >= 8 else '-1' for k in df_data['quality']]
    X = df_data[['alcohol', 'pH']].values
    data = p.fit(X, y)

    numvar = X.shape[0]
    x1_ = []
    y1_ = []
    x2_ = []
    y2_ = [] 
    for i in data:
        x1 = i[0]
        y1 = i[1]
        
        x1_.append(x1)
        y1_.append(y1)
        if i[1] == 0 :
            for j in range(numvar):
                x2 = X[j]
                slop = -i[2][0] / i[2][1]
                y2 = slop * x2
                x2_.append(x2)
                y2_.append(y2)
            break

    print(x2_)
    print(y2_)
    plt.subplot(121)
    plt.plot(x1_, y1_, color='red')
    plt.xlabel('epoch')
    plt.ylabel('errors')
    plt.title("Errors as a function of epoch")

    plt.subplot(122)
    color = ['blue' if k >= good_thresh else 'red' for k in df_data['quality']]
    size = [10 if k >= good_thresh else 10 if k <= bad_thresh else 0 for k in df_data['quality']]
    plt.scatter(df_data['alcohol'], df_data['pH'], c = color, s = size)
    plt.plot(x2_, y2_)
    plt.xlabel('alcohol')
    plt.ylabel('pH')
    plt.title("Decision boundary on epoch:")
    plt.show()
plot_performance(p, 'winequality-red.csv', 8, 3, epoch = 100, save_plot=False)

