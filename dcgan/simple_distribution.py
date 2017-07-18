import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.stats import norm

mpl.use('Agg')
plt.style.use('bmh')

# Plot norm distribution PDF
np.random.seed(0)
X = np.arange(-3, 3, 0.001)
Y = norm.pdf(X, 0, 1)
fig = plt.figure()
plt.plot(X, Y)
plt.tight_layout()
plt.savefig("normal-pdf.png")

# Plot random sample
samples = 35
X = np.random.normal(0, 1, samples)
Y = np.zeros(samples)
fig = plt.figure(figsize=(7, 3))
plt.scatter(X, Y, color='k')
plt.xlim((-3,3))
frame = plt.gca()
frame.axes.get_yaxis().set_visible(False)
plt.savefig("normal-samples.png")
