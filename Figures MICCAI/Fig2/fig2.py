import numpy as np
import matplotlib.pylab as plt 
import seaborn as sns
import pandas as pd
from scipy.stats import expon, ttest_ind

linear = pd.read_csv("EdgeError_Linear.csv", delimiter=',', header=None).values[0]
nonlinear = pd.read_csv("EdgeError_NonLinear.csv", delimiter=',', header=None).values[0]
data = np.stack((linear, nonlinear), axis=1)
_, linear_lambda = expon.fit(linear)
_, nonlinear_lambda = expon.fit(nonlinear)
print("Linear: ", linear_lambda)
print("NonLinear: ", nonlinear_lambda)

T,Tp = ttest_ind(linear, nonlinear, equal_var=False)
print("T-test: ", T, Tp)

x = np.linspace(0, 1.8)
colors = ['cornflowerblue', 'tomato']
exp1 = (1/linear_lambda)*np.exp(-(1/linear_lambda)*x)
exp2 = (1/nonlinear_lambda)*np.exp(-(1/nonlinear_lambda)*x)

# Fig1A
fig, ax = plt.subplots(figsize=(6,4))
right_side, top_side = ax.spines["right"], ax.spines["top"]
right_side.set_visible(False), top_side.set_visible(False)
plt.ylabel("Density")
plt.xlabel("MAE")
hist = plt.hist(data, histtype='bar', color=colors, label=['linear', 'non-linear'], density=True, alpha=0.7)
exp_linear = ax.plot(x, exp1, '--', color='tomato', lw=1.5)
exp_nonlinear = ax.plot(x, exp2, '--', color='cornflowerblue', lw=1.5)
plt.legend(loc='upper right', frameon=False, fontsize=10)
fig.savefig("EdgeExponentialFit.png", format='png', dpi=1200)
