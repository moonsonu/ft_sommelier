import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_scatter_matrix(wine_data, good_threshold, bad_threshold, save_plot=False):
	red_wine = pd.read_csv(wine_data, sep=';')
	numvars, numdata = red_wine.shape
	print('numvars: {0} numdata: {1}'.format(numvars, numdata))
	fig = plt.figure()
	fig, axes = plt.subplots(numdata, numdata, figsize=(50, 50))
	for i in range(numdata):
		for j in range(numdata):
			if (i == j):
				axes[i, j].text(0.5, 0.5, red_wine.columns[i], ha='center')
			else:
				colors = ['blue' if k >= good_threshold else 'red' for k in red_wine['quality']]
				axes[i, j].scatter(red_wine[red_wine.columns[i]], red_wine[red_wine.columns[j]], c=colors)
	if save_plot=True
	   plt.savefig("./wine_data.png")
	plt.tight_layout()
	fig
