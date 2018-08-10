import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
    df = pd.read_csv(wine_data, sep=';')
    numvars, numdata = df.shape
    fig, axes = plt.subplots(numdata, numdata, figsize=(50,50))
    for i in range(numdata):
        for j in range(numdata):
            if i == j:
                df.columns[i]
                axes[i, j].text(0.5, 0.5, df.columns[i], ha='center')
            else:
                for k in df['quality']
                    color = ['green' if k == good_threshold 'red' elif k == bad_threshold else 'none']
                axes[i,j].scatter(df[df.columns[i]], df[df.columns[j]], c = color)
    if save_plot == True:
        plt.savefig("./wine_data.png")
    plt.tight_layout()
    fig

plot_scatter_matrix('winequality-red.csv', 6, 5, save_plot=False)
