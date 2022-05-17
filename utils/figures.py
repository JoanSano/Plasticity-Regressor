import numpy as np
import matplotlib.pylab as plt
from scipy.stats import probplot

def boxplot(png_path, args, mse, mse_z, mae_z, pcc_z, cs_z, kl_z, js_z, PAT_subjects):
    subject_to_follow_max = np.argmax(mse) # Highest zscore of reconstruction error
    subject_to_follow_min = np.argmin(mse) # Lowest zscore of reconstruction error
    fig, ax = plt.subplots(figsize=(8,6))
    positions = [2, 4, 6, 8, 10, 12]
    bx_data = np.array([mse_z, mae_z, pcc_z, cs_z, kl_z, js_z]).T
    for i in range(6):   
        xdata = positions[i]+0*bx_data[:,i]+np.random.normal(0,0.15,size=bx_data[:,i].shape)
        ax.scatter(xdata, bx_data[:,i], s=15)
        if i==0:
            ax.plot(xdata[subject_to_follow_max], bx_data[subject_to_follow_max,i], 'k*', markersize=10, label=PAT_subjects[subject_to_follow_max])
            ax.plot(xdata[subject_to_follow_min], bx_data[subject_to_follow_min,i], 'ks', markersize=7, label=PAT_subjects[subject_to_follow_min])
        else:
            ax.plot(xdata[subject_to_follow_max], bx_data[subject_to_follow_max,i], 'k*', markersize=10)
            ax.plot(xdata[subject_to_follow_min], bx_data[subject_to_follow_min,i], 'ks', markersize=7)
    bx = ax.boxplot(bx_data, 
        positions=positions, widths=1.5, patch_artist=True,
        showmeans=False, showfliers=False,
        medianprops={'color':'black', 'linewidth':1.8},
        #meanprops={'marker':'s', 'markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':6}
    )
    ax.set_ylabel('zscore', fontsize=16), ax.set_xticklabels(['MSE', 'MAE', 'PCC', 'CS', 'KL', 'JS'], fontsize=16)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    for b in bx['boxes']:
        b.set_edgecolor('k') # or try 'black'
        b.set_facecolor([0.3,0.3,0.6,0.2])
        b.set_linewidth(1.5)
    plt.legend(loc=9, frameon=True, fontsize=10, ncol=2, bbox_to_anchor=(0.5,1.1))
    plt.savefig(png_path+args.model+'_boxplot.png', dpi=900)
    plt.savefig(png_path+args.model+'_boxplot.eps', dpi=900)

def normality_plots(png_path, mse, mae, pcc, cs, kl, js, args, PAT_subjects):
    fig, ax = plt.subplots(figsize=(6,4))
    norm_mse, fit_mse = probplot(mse)
    norm_mae, fit_mae = probplot(mae)
    norm_pcc, fit_pcc = probplot(pcc)
    norm_cs, fit_cs = probplot(cs)
    norm_kl, fit_kl = probplot(kl)
    norm_js, fit_js = probplot(js)
    
    ax.scatter(norm_mse[0], norm_mse[1], s=15, label='MSE')
    ax.scatter(norm_mae[0], norm_mae[1], s=15, label='MAE')
    ax.scatter(norm_pcc[0], norm_pcc[1], s=15, label='PCC')
    ax.scatter(norm_cs[0], norm_cs[1], s=15, label='CS')
    ax.scatter(norm_kl[0], norm_kl[1], s=15, label='KL')
    ax.scatter(norm_js[0], norm_js[1], s=15, label='JS')
    
    ax.set_title(""), ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    plt.savefig(png_path+args.model+'_norm_mse.png', dpi=900)
    plt.savefig(png_path+args.model+'_norm_mse.eps', dpi=900)