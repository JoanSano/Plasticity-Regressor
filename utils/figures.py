import numpy as np
import matplotlib.pylab as plt
from scipy.stats import probplot, pearsonr, permutation_test, ttest_ind, mannwhitneyu, kruskal

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
    fig, ax = plt.subplots(figsize=(6,4.5))
    norm_mse, fit_mse = probplot(mse)
    norm_mae, fit_mae = probplot(mae)
    norm_pcc, fit_pcc = probplot(pcc)
    norm_cs, fit_cs = probplot(cs)
    norm_kl, fit_kl = probplot(kl)
    norm_js, fit_js = probplot(js)
    
    ax.scatter(norm_mse[0], norm_mse[1], s=10, label='MSE')
    ax.scatter(norm_mae[0], norm_mae[1], s=10, label='MAE')
    ax.scatter(norm_pcc[0], norm_pcc[1], s=10, label='PCC')
    ax.scatter(norm_cs[0], norm_cs[1], s=10, label='CS')
    ax.scatter(norm_kl[0], norm_kl[1], s=10, label='KL')
    ax.scatter(norm_js[0], norm_js[1], s=10, label='JS')

    ax.plot(np.arange(-2,2,0.01), np.arange(-2,2,0.01), 'k--', linewidth=1)
    
    ax.set_title(""), ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xlabel('Theoretical Quantiles', fontsize=12), ax.set_ylabel('zscore', fontsize=12)
    ax.set_xticks([-2,-1,0,1,2]), ax.set_xticklabels(['-2', '-1', '0', '1', '2'])
    plt.legend(loc=2, frameon=False, fontsize=10, ncol=2)
    plt.savefig(png_path+args.model+'_normality.png', dpi=900)
    plt.savefig(png_path+args.model+'_normality.eps', dpi=900)

    print("Linear fits for the normality plots:")
    print("MSE: r = ",fit_mse[2])
    print("MAE: r = ",fit_mae[2])
    print("PCC: r = ",fit_pcc[2])
    print("CS: r = ",fit_cs[2])
    print("KL: r = ",fit_kl[2])
    print("JS: r = ",fit_js[2])
    print("=====================================")

def size_correlation(figs_path, args, mae, pcc, tumor_sizes, PAT_subjects):
    from models.methods import to_array

    #################################
    ### Correlation size vs error ###
    #################################

    tm_size = to_array(tumor_sizes)
    # Dropping 3 biggest
    tm_3drop = tm_size[tm_size.argsort()[:-3][::-1]]
    mae_3drop = mae[tm_size.argsort()[:-3][::-1]]
    pcc_3drop = pcc[tm_size.argsort()[:-3][::-1]]
    # Dropping 4 biggest
    tm_4drop = tm_size[tm_size.argsort()[:-4][::-1]]
    mae_4drop = mae[tm_size.argsort()[:-4][::-1]]
    pcc_4drop = pcc[tm_size.argsort()[:-4][::-1]]

    r_mae, p_mae = pearsonr(mae, tm_size)
    r_pcc, p_pcc = pearsonr(pcc, tm_size)
    r_mae_3drop, p_mae_3drop = pearsonr(mae_3drop, tm_3drop)
    r_pcc_3drop, p_pcc_3drop = pearsonr(pcc_3drop, tm_3drop)
    r_mae_4drop, p_mae_4drop = pearsonr(mae_4drop, tm_4drop)
    r_pcc_4drop, p_pcc_4drop = pearsonr(pcc_4drop, tm_4drop)
    p_mae, p_pcc, p_mae_3drop, p_pcc_3drop, p_mae_4drop, p_pcc_4drop = p_mae/2, p_pcc/2, p_mae_3drop/2, p_pcc_3drop/2, p_mae_4drop/2, p_pcc_4drop/2

    # Permutation tests 
    samples = 500
    statistic = lambda x, y: pearsonr(x,y)[0]
    permu_pcc = permutation_test((pcc, tm_size), statistic, n_resamples=samples, alternative='less')
    permu_3pcc = permutation_test((pcc_3drop, tm_3drop), statistic, n_resamples=samples, alternative='less')
    permu_4pcc = permutation_test((pcc_4drop, tm_4drop), statistic, n_resamples=samples, alternative='less')
    permu_mae = permutation_test((mae, tm_size), statistic, n_resamples=samples, alternative='greater')
    permu_3mae = permutation_test((mae_3drop, tm_3drop), statistic, n_resamples=samples, alternative='greater')
    permu_4mae = permutation_test((mae_4drop, tm_4drop), statistic, n_resamples=samples, alternative='greater')
    
    fig, ax = plt.subplots(figsize=(5,3.5))
    plt.scatter(tm_size, pcc, s=10, label='r = ' + str(round(r_pcc,3)))
    plt.scatter(tm_3drop, pcc_3drop, s=10, label='r = ' + str(round(r_pcc_3drop,3)))

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_ylim([0.82,0.905]), ax.set_yticks([0.82,0.84,0.86,0.88,0.90]), ax.set_yticklabels(['0.82','0.84','0.86','0.88','0.90'], fontsize=8)
    ax.set_xticks([0,20,40,60,80]), ax.set_xticklabels(['0','20','40','60','80'], fontsize=8)
    ax.set_xlabel('Tumor size (cm$^3$)', fontsize=8), ax.set_ylabel('PCC', fontsize=8)
    plt.legend(loc=4, frameon=True, fontsize=8)
    plt.savefig(figs_path+args.model+'_size-effects.png', dpi=900)
    plt.savefig(figs_path+args.model+'_size-effects.eps', dpi=900)

    print("Correlations with tumor size:")
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f}".format(r_pcc, p_pcc, permu_pcc.pvalue))
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (3 dropped)".format(r_pcc_3drop, p_pcc_3drop, permu_3pcc.pvalue))
    print("PCC: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (4 dropped)".format(r_pcc_4drop, p_pcc_4drop, permu_4pcc.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f}".format(r_mae, p_mae, permu_mae.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (3 dropped)".format(r_mae_3drop, p_mae_3drop, permu_3mae.pvalue))
    print("MAE: r = {:.4f}, one-sided p = {:.4f} and p_permu = {:.4f} (4 dropped)".format(r_mae_4drop, p_mae_4drop, permu_4mae.pvalue))
    print("=============================")

    #################################
    ### Tumor size between groups ###
    #################################

def type_effects(figs_path, args, mae, pcc, tumor_types, PAT_subjects, alpha=0.05):
    from models.methods import f_test
    meningioma, glioma = [] , []
    for s in range(len(PAT_subjects)):
        if 'gioma' in tumor_types[PAT_subjects[s]]:
            meningioma.append([pcc[s], mae[s]])
        else:
            glioma.append([pcc[s], mae[s]])
    meningioma = np.array(meningioma, dtype=np.float64)
    glioma = np.array(glioma, dtype=np.float64)

    mean_pcc_menin, mean_mae_menin = np.mean(meningioma[:,0]), np.mean(meningioma[:,1])
    mean_pcc_gliom, mean_mae_gliom = np.mean(glioma[:,0]), np.mean(glioma[:,1])
    std_pcc_menin, std_mae_menin = np.std(meningioma[:,0])/len(meningioma), np.std(meningioma[:,1])/len(meningioma)
    std_pcc_gliom, std_mae_gliom = np.std(glioma[:,0])/len(glioma), np.std(glioma[:,1])/len(glioma)
    
    # PCC
    _, p_var = f_test(meningioma[:,0], glioma[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(meningioma[:,0], glioma[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(meningioma[:,0], glioma[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(meningioma[:,1], glioma[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(meningioma[:,1], glioma[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(meningioma[:,1], glioma[:,1], alternative='two-sided')

    print("Error with tumor type: (MEAN +- SEM)")
    print("Meningioma: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_menin, std_pcc_menin, mean_mae_menin, std_mae_menin))
    print("Glioma: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_gliom, std_pcc_gliom, mean_mae_gliom, std_mae_gliom))
    print("Differences between tumor types, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor types, Mann-Whitney:")
    print("PCC one-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE one-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar([1,2], [mean_pcc_menin, mean_pcc_gliom], 
        yerr=[std_pcc_menin, std_pcc_gliom],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.855,0.885]), ax.set_yticks([0.86,0.87,0.88]), ax.set_yticklabels(['0.86','0.87','0.88'])
    ax.set_xticks([1,2]), ax.set_xticklabels(['Meningioma', 'Glioma']), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-type.png', dpi=900)
    plt.savefig(figs_path+args.model+'_tumor-type.eps', dpi=900)

def location_effects(figs_path, args, mae, pcc, tumor_locs, PAT_subjects, alpha=0.05):
    from models.methods import f_test
    frontal, other = [] , []
    for s in range(len(PAT_subjects)):
        if 'frontal' in tumor_locs[PAT_subjects[s]].lower():
            frontal.append([pcc[s], mae[s]])
        else:
            other.append([pcc[s], mae[s]])
    frontal = np.array(frontal, dtype=np.float64)
    other = np.array(other, dtype=np.float64)

    mean_pcc_front, mean_mae_front = np.mean(frontal[:,0]), np.mean(frontal[:,1])
    mean_pcc_oth, mean_mae_oth = np.mean(other[:,0]), np.mean(other[:,1])
    std_pcc_front, std_mae_front = np.std(frontal[:,0])/len(frontal), np.std(frontal[:,1])/len(frontal)
    std_pcc_oth, std_mae_oth = np.std(other[:,0])/len(other), np.std(other[:,1])/len(other)
    
    # PCC
    _, p_var = f_test(frontal[:,0], other[:,0])
    eq_var = True if p_var>alpha else False
    _, p_pcc_T = ttest_ind(frontal[:,0], other[:,0], equal_var=eq_var, alternative='two-sided')
    _, p_pcc_U = mannwhitneyu(frontal[:,0], other[:,0], alternative='two-sided')
    # MAE
    _, p_var = f_test(frontal[:,1], other[:,1])
    eq_var = True if p_var>alpha else False
    _, p_mae_T = ttest_ind(frontal[:,1], other[:,1], equal_var=eq_var, alternative='two-sided')
    _, p_mae_U = mannwhitneyu(frontal[:,1], other[:,1], alternative='two-sided')

    print("Error with tumor location: (MEAN +- SEM)")
    print("Frontal: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_front, std_pcc_front, mean_mae_front, std_mae_front))
    print("Other: PCC = {:.4f} +/- {:.4}, MAE = {:.4f} +/- {:.4}".format(mean_pcc_oth, std_pcc_oth, mean_mae_oth, std_mae_oth))
    print("Differences between tumor locations, T-test:")
    print("PCC two-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_T, p_pcc_T/2))
    print("MAE two-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_T, p_mae_T/2))
    print("Differences between tumor locations, Mann-Whitney:")
    print("PCC one-sided p = {:.4f} and one-sided p = {:.4}".format(p_pcc_U, p_pcc_U/2))
    print("MAE one-sided p = {:.4f} and one-sided p = {:.4}".format(p_mae_U, p_mae_U/2))
    print("=============================")

    fig, ax = plt.subplots(figsize=(5,4))
    ax.bar([1,2], [mean_pcc_front, mean_pcc_oth], 
        yerr=[std_pcc_front, std_pcc_oth],linewidth=2,
        color=[0,0,1,0.5],edgecolor=[0,0,0,1],error_kw=dict(lw=2),
        ecolor='k', capsize=15, width=0.75, align='center'
    )

    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
    ax.set_ylim([0.855,0.885]), ax.set_yticks([0.86,0.87,0.88]), ax.set_yticklabels(['0.86','0.87','0.88'])
    ax.set_xticks([1,2]), ax.set_xticklabels(['Frontal', 'Other']), ax.set_ylabel('PCC')
    plt.savefig(figs_path+args.model+'_tumor-loc.png', dpi=900)
    plt.savefig(figs_path+args.model+'_tumor-loc.eps', dpi=900)