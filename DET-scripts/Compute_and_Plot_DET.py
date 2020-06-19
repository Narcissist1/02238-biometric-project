import math
import numpy as np
from matplotlib import pyplot as plt
# install using:
# pip install numpy matplotlib
# or
# conda install numpy matplotlib

def compute_and_plot_det(true_scores, false_scores):
	Pfa, Pmiss = compute_det(true_scores, false_scores)
	plot_det(Pfa, Pmiss, "algorithm", True)

def plot_det(Pfa, Pmiss, evaluation_type="algorithm", abbreviate_labels=True):
	x_ticks = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
	x_ticklabels = np.array(['0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '5', '20', '40',])
	y_ticks = np.array([1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
	y_ticklabels = np.array(['0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '5', '20', '40'])
	plt.plot(Pfa, Pmiss, label="example")
	plt.xscale("log")
	plt.yscale("log")
	xmin, xmax, ymin, ymax = plt.axis()
	plt.xticks(x_ticks, x_ticklabels)
	plt.yticks(y_ticks, y_ticklabels)
	plt.xlim(10 ** math.floor(math.log(xmin, 10)), 0.5)
	plt.ylim(10 ** math.floor(math.log(ymin, 10)), 0.5)
	x_label, y_label = get_axes_labels(evaluation_type, abbreviate_labels)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.grid(True)
	plt.legend(loc=0)
	plt.savefig("example.png", format="png", bbox_inches="tight")
	plt.cla()
	plt.clf()
	plt.close()

def compute_det(true_scores, false_scores):
	num_true = true_scores.shape[0]
	num_false = false_scores.shape[0]
	assert num_true > 0, "Vector of target scores is empty"
	assert num_false > 0, "Vector of nontarget scores is empty"

	total = num_true + num_false

	Pmiss = np.zeros((total + 1))
	Pfa = np.zeros((total + 1))

	scores = np.zeros((total, 2))
	scores[:num_false, 0] = false_scores
	scores[:num_false, 1] = 0
	scores[num_false:, 0] = true_scores
	scores[num_false:, 1] = 1

	scores = __DETsort__(scores)

	sumtrue = np.cumsum(scores[:, 1], axis=0)
	sumfalse = num_false - (np.arange(1, total + 1) - sumtrue)

	Pmiss[0] = 0
	Pfa[0] = 1
	Pmiss[1:] = sumtrue / num_true
	Pfa[1:] = sumfalse / num_false
	return Pfa, Pmiss

def __DETsort__(x, col=''):
	assert x.ndim > 1, 'x must be a 2D matrix'
	if col == '':
		list(range(1, x.shape[1]))

	ndx = np.arange(x.shape[0])
	ind = np.argsort(x[:, 1], kind='mergesort')
	ndx = ndx[ind]
	ndx = ndx[::-1]
	ind = np.argsort(x[ndx, 0], kind='mergesort')

	ndx = ndx[ind]
	sort_scores = x[ndx, :]
	return sort_scores

def get_axes_labels(evaluation_type="algorithm", abbreviate_labels=True):
	if evaluation_type == 'algorithm':
		if abbreviate_labels:
			x_label = 'FMR (in %)'
			y_label = 'FNMR (in %)'
		else:
			x_label = 'False Match Rate (in %)'
			y_label = 'False Non-Match Rate (in %)'
	elif evaluation_type == 'system':
		if abbreviate_labels:
			x_label = 'FAR (in %)'
			y_label = 'FRR (in %)'
		else:
			x_label = 'False Acceptance Rate (in %)'
			y_label = 'False Rejection Rate (in %)'
	elif evaluation_type == 'PAD':
		if abbreviate_labels:
			x_label = 'APCER (in %)'
			y_label = 'BPCER (in %)'
		else:
			x_label = 'Attack Presentation Classification Error Rate (in %)'
			y_label = 'Bona Fide Presentation Classification Error Rate (in %)'
	elif evaluation_type == 'identification':
		if abbreviate_labels:
			x_label = 'FPIR (in %)'
			y_label = 'FNIR (in %)'
		else:
			x_label = 'False Positive Identification Rate (in %)'
			y_label = 'False Negative Identification Rate (in %)'
	else:
		x_label = 'Type I Error Rate (in %)'
		y_label = 'Type II Error Rate (in %)'
	return x_label, y_label