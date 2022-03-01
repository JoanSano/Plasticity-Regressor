import numpy as np
import matplotlib.pylab as plt 
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

# Fig1A
linear = pd.read_csv("LinearGraphError.csv", delimiter=',', header=None).values[0]
nonlinear = pd.read_csv("NonLinearGraphError.csv", delimiter=',', header=None).values[0]
data = np.stack((linear, nonlinear), axis=1)
linear_mean, linear_std = np.mean(linear), np.std(linear)
nonlinear_mean, nonlinear_std = np.mean(nonlinear), np.std(nonlinear)
t,p = ttest_ind(linear, nonlinear, equal_var=False)
print("Linear: ", linear_mean, "pm", linear_std)
print("NonLinear: ", nonlinear_mean, "pm", nonlinear_std)
print("t stat: ", t, "p-val: ", p)
colors = ['cornflowerblue', 'tomato']
lims = (0.20, 0.45)
linear_limits = np.linspace(linear_mean-linear_std, linear_mean+linear_std)
nonlinear_limits = np.linspace(nonlinear_mean-nonlinear_std, nonlinear_mean+nonlinear_std)

fig, (ax1, ax2) = plt.subplots(figsize=(6,4),  nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 10]})
# Top
right_side, top_side, bottom_side, left_side = ax1.spines["right"], ax1.spines["top"], ax1.spines["bottom"], ax1.spines["left"]
right_side.set_visible(False), top_side.set_visible(False), bottom_side.set_visible(False), left_side.set_visible(False)
lin_std = ax1.plot(linear_limits, np.ones((50,)), linewidth=1, color='black')
lin_mean = ax1.plot(linear_mean, 1, '*', linewidth=2, color='cornflowerblue')
nonlin_std = ax1.plot(nonlinear_limits, np.ones((50,))+1, linewidth=1, color='black')
nonlin_mean = ax1.plot(nonlinear_mean, 2, '*', linewidth=2, color='tomato')
ax1.set_yticklabels([]), ax1.set_xticks([]), ax1.set_yticks([])
ax1.set_xlim(lims), ax1.set_ylim((0.75,2.25))

# Bottom
right_side, top_side = ax2.spines["right"], ax2.spines["top"]
right_side.set_visible(False), top_side.set_visible(False)
plt.ylabel("Counts")
plt.xlabel("MAE")
hist = plt.hist(data, histtype='bar', color=colors, label=['linear', 'non-linear'], bins=8)
plt.axvline(linear_mean, color='black', linestyle='dashed', linewidth=1)
plt.axvline(nonlinear_mean, color='black', linestyle='dashed', linewidth=1)
ax2.set_xlim(lims), ax2.set_yticks([0, 1, 2, 3, 4]), ax2.set_yticklabels([0, 1, 2, 3, 4])
plt.legend(loc='upper right', frameon=False, fontsize=10)
fig.savefig("ReconstructionError.png", format='png', dpi=1200)

# Fig 1B Training/Validation Loss
linearTR = pd.read_csv("TrainingLossLinear.csv").values[:125,:2]
linearVL = pd.read_csv("ValidationLossLinear.csv").values[:125,:2]
nonlinearTR = pd.read_csv("TrainingLossNonLinear.csv").values[:125,:2]
nonlinearVL = pd.read_csv("ValidationLossNonLinear.csv").values[:125,:2]

fig, ax = plt.subplots(figsize=(6,4))
right_side, top_side = ax.spines["right"], ax.spines["top"]
right_side.set_visible(False), top_side.set_visible(False)
plt.xlabel("Epoch")
plt.ylabel("MSE")
lintrain = plt.plot(linearTR[:,0],linearTR[:,1], color='cornflowerblue', linewidth=2)
linval = plt.plot(linearVL[:,0],linearVL[:,1], '--', color='cornflowerblue', linewidth=2)
nonlintrain = plt.plot(nonlinearTR[:,0],nonlinearTR[:,1], color='tomato', linewidth=2)
nonlinval = plt.plot(nonlinearVL[:,0],nonlinearVL[:,1], '--', color='tomato', linewidth=2)
lab1 = plt.plot(1000,1,color='black', linewidth=2, label='Training')
lab2 = plt.plot(1001,1,'--',color='black', linewidth=2, label='Validation')
ax.set_xlim((0, 125))
plt.legend(loc='upper right', frameon=False, fontsize=10)
fig.savefig("TrainingError.png", format='png', dpi=1200)
