import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC

X, Y = make_blobs(n_samples=70, centers=2, random_state=0, cluster_std=0.7)

plt.scatter(X[:,0],X[:,1],c=Y)

plt.savefig('scatterplot.png')

svm_model = SVC(kernel='linear', C=1E10)
svm_model.fit(X,Y)


def plot_decision_function(model):
	ax = plt.gca()
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()

	x = np.linspace(xlim[0], xlim[1], 30)
	y = np.linspace(ylim[0], ylim[1], 30)
	Y, X = np.meshgrid(y,x)
	xy = np.vstack([X.ravel(), Y.ravel()]).T
	P = model.decision_function(xy).reshape(X.shape)
	ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
	ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

plot_decision_function(svm_model)
plt.savefig('scatterplot_with_decision_function.png')












