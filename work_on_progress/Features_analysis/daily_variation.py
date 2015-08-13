# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:36:56 2015

@author: ajaver
"""

import pandas as pd
import numpy as np
import statsmodels.stats.multitest as smm
import matplotlib.pylab as plt
#rank sum test
from scipy.stats import ranksums, ttest_ind

def getPValues(feat_mean, strain_list, feat_list):    
    strain_groups = feat_mean.groupby('Strain');
    features_N2 = strain_groups.get_group('N2');
    
    pvalue_table = pd.DataFrame(np.nan, index = feat_list, columns = strain_list, dtype = np.float64)
    for strain in pvalue_table.columns.values:
        features_S = strain_groups.get_group(strain);
        for feat in pvalue_table.index.values:
            x, y = features_N2[feat].values, features_S[feat].values
            dd, p_value = ttest_ind(x,y, equal_var=False)
            #dd, p_value = ranksums(x,y)
            
            #p_value positive if N2 is larger than the strain
            pvalue_table.loc[feat, strain] = p_value
        
        good = ~np.isnan(pvalue_table[strain])
        #correct for false discovery rate using 2-stage Benjamini-Krieger-Yekutieli
        reject, pvals_corrected, alphacSidak, alphacBonf = \
        smm.multipletests(pvalue_table.loc[good,strain].values, method = 'fdr_tsbky')
        pvalue_table.loc[good,strain] = pvals_corrected
    return pvalue_table

def getZStats(feat_mean, strain, strain_ref = 'N2'):
    #%%
    strain_index = (feat_mean['Strain']==strain)
    N2_index = (feat_mean['Strain']==strain_ref);
    
    good = strain_index | N2_index
    subfeat = feat_mean[good]
    strain_names =  subfeat['Strain'].copy()
    
    del subfeat['Strain']
    nan_feat = subfeat.isnull().any()
    nan_feat = nan_feat | (subfeat==0).all()
    subfeat = subfeat.loc[:,~nan_feat]
    
    subfeat_z = (subfeat - subfeat.mean())/subfeat.std()
    subfeat_z['Strain'] = strain_names
    
    z_stats = subfeat_z.groupby('Strain').agg([np.mean, np.std, 'count'])
    z_stats = z_stats.transpose()
    
    idx = pd.IndexSlice
    
    z_mean = z_stats.loc[(idx[:,'mean']), :]
    z_mean.index = z_mean.index.droplevel(1)

    z_std = z_stats.loc[(idx[:,'std']), :]
    z_std.index = z_std.index.droplevel(1)
    
    z_count = z_stats.loc[(idx[:,'count']), :]
    z_count.index = z_count.index.droplevel(1)
    
    z_err = z_std/np.sqrt(z_count)
    #%%
    dd = {'mean':z_mean[strain], 'mean_ref':z_mean[strain_ref],
           'err':z_err[strain], 'err_ref':z_err[strain_ref]}
    z_stats = pd.DataFrame(data=dd)
    #%%
    z_values = {}

    dd = subfeat_z[subfeat_z['Strain'] == strain]
    dd = dd.drop('Strain', axis = 1)
    z_values['feat'] = np.tile(dd.columns.values, (len(dd), 1)).flatten()
    z_values['values'] = dd.values.flatten()
    
    dd = subfeat_z[subfeat_z['Strain'] == strain_ref]
    dd = dd.drop('Strain', axis = 1)
    z_values['feat_ref'] = np.tile(dd.columns.values, (len(dd), 1)).flatten()
    z_values['values_ref'] = dd.values.flatten()
    #%%
    return z_stats, nan_feat, z_values

def plotZStats(z_stats, p_values, z_values, save_name):
    #%%
    
    ord_p = np.abs(np.log(p_values))
    ord_p[z_stats['mean']<z_stats['mean_ref']] *= -1
    z_stats['ord_p'] = ord_p
    z_stats['p_values'] = p_values
    z_stats.sort('ord_p', inplace=True)
    
    feat_names = z_stats.index
    feat_ids = {}
    for ii, feat in enumerate(feat_names):
        feat_ids[feat] = ii
    
    z_values['feat_id'] = [feat_ids[feat] for feat in z_values['feat']]
    z_values['feat_id_ref'] = [feat_ids[feat] for feat in z_values['feat_ref']]
    
    x = np.arange(len(z_stats))
    
    #%%
    fig = plt.figure()
    
    fig.set_size_inches([6, 45])
    plt.title(save_name)    
    plt.plot(z_values['values_ref'], z_values['feat_id_ref'], '.', color = '#87CEEB')
    plt.plot(z_values['values'], z_values['feat_id'], '.', color = "#90EE90")
    
    plt.errorbar(z_stats['mean_ref'], x, xerr = z_stats['err_ref'], fmt = 'ob')
    plt.errorbar(z_stats['mean'], x, xerr = z_stats['err'], fmt = 'og')
    plt.xlim([-2.5, 2.5])
    plt.ylim([x[0]-2, x[-1]+2])
    
    #show valid limits
    bad_p = np.where(z_stats['p_values']>0.05)[0]
    lim1_p = bad_p[0]-1
    lim2_p = bad_p[-1]+1
    limX = plt.xlim()
    plt.plot(limX, [lim1_p, lim1_p], '--r')
    plt.plot(limX, [lim2_p, lim2_p], '--r')
       
    
    plt.yticks(x, feat_names)
    ax = plt.gca()
    ax.tick_params(axis='y', which='major', labelsize=5, gridOn=True)
    plt.tight_layout()
    plt.savefig(save_name, dpi=1200)
    
    plt.close()
#%%
    
if __name__ == '__main__':
    #plates_file = '/Users/ajaver/Desktop/Gecko_compressed/Results/PlateFeatures.hdf5'
    plates_file = '/Volumes/behavgenom$/GeckoVideo/Ana_Strains/PlateFeatures.hdf5'
    #plates_file = '/Volumes/behavgenom$/GeckoVideo/Ana_Strains/PlateFeatures_MED.hdf5'
    with pd.HDFStore(plates_file, 'r') as plates_fid:
        feat_mean = plates_fid['/avg_feat_per_plate']
        video_feat = plates_fid['/video_features']
    
    feat_mean.index = feat_mean['Base_Name'].values
    del feat_mean['Base_Name']
    
    video_feat.index = video_feat['Base_Name'].values
    del video_feat['Base_Name']
    
    dates = [x.split('_')[2] for x in video_feat.index]
    dates = [x if x != '16062015' else '15062015' for x in dates ]
    video_feat['Dates'] = dates
    feat_mean['Dates'] = dates
    
    strain_list = [x for x in feat_mean['Strain'].unique() if x != 'N2']
    
    
    #filter for only the main feat_mean.columns (no subdivision)
    feat_list = [feat for feat in feat_mean.columns if not any(x in feat \
    for x in ['_Pos', '_Neg', '_Abs', '_Backward', '_Foward', '_Paused'])]
    feat_mean = feat_mean.loc[:, feat_list]
    
    feat_list = [feat for feat in feat_mean.columns if not any(x in feat for x in ['Strain', 'Dates'])]
    feat_list = sorted(feat_list)
    
    #%%
    tot_rows = len(feat_list)*len(feat_mean)
    feat_col = np.zeros(tot_rows, np.dtype(('U',50)))
    date_col = np.zeros(tot_rows, np.dtype(('U',6)))
    strain_col = np.zeros(tot_rows, np.dtype(('U',20)))
    value_col = np.zeros(tot_rows, np.float)
    z_col  = np.zeros(tot_rows, np.float)
    
    tot_plates = len(feat_mean.index)
    for ii, feat in enumerate(feat_list):
        bot = ii*tot_plates
        top = (ii+1)*tot_plates
        feat_col[bot:top] = feat
        date_col[bot:top] = feat_mean['Dates']
        strain_col[bot:top] = feat_mean['Strain']
        value_col[bot:top] = feat_mean[feat]        
        
        dd = feat_mean[feat]
        dd = (dd - np.nanmean(dd))/np.nanstd(dd)
        z_col[bot:top] = dd        
    
    dat_lab = pd.DataFrame({'Features': feat_col, 'Dates': date_col, 'Values': value_col, 'Z_Values': z_col, 'Strain' : strain_col})
    #%%
    import seaborn as sns
    
    for strain in feat_mean['Strain'].unique(): 
        strain_data = dat_lab[dat_lab['Strain']==strain]    
        
        sns.set_context("paper")
        plt.figure(figsize=(6, 36))
        
        sns.boxplot(y = 'Features', x = 'Z_Values', hue = 'Dates', data = strain_data)
        plt.title(strain)
    #%%
    #feat_mean.boxplot('Primary_Wavelength_Foward', by='Strain')
#    
#    #%%
#
#    good = (video_feat['Total_Frames']>170000) &  (video_feat['Total_Frames']<200000)
#    #good =  good & (video_feat['Dates'] != '19062015')&
#    video_feat = video_feat[good]
#    feat_mean = feat_mean[good]
#
#    
#
#    all_pvalues = {}
#    all_pvaluesC = {}
#    for strain in strain_list:
#        all_pvalues[strain] = pd.DataFrame()
#        all_pvaluesC[strain] = pd.DataFrame()
#    #%%
#    date_list = list(video_feat['Dates'].unique()) +  ['All']
#    for ii_date, date in enumerate(date_list):
#        if date != 'All':       
#            good =  (video_feat['Dates'] == date)
#            video_feat_sub = video_feat[good]
#            feat_mean_sub = feat_mean[good]
#        else:
#            video_feat_sub = video_feat
#            feat_mean_sub = feat_mean
#        
#        tot_samples = feat_mean_sub['Strain'].value_counts()
#        print(date)
#        print(tot_samples)
#    #%%
#        pvalue_table = getPValues(feat_mean_sub, strain_list, feat_list)
#        
#        save_prefix = 'zstat_'
#        for ii_strain, strain in enumerate(strain_list):
#        #%%
#            z_stats, nan_feat, z_values = getZStats(feat_mean_sub, strain, strain_ref = 'N2')
#            
#            p_values = pvalue_table.loc[~nan_feat, strain]
#            all_pvalues[strain][date] = p_values
#            
#            # correct for false discovery rate using 2-stage Benjamini-Krieger-Yekutieli
#            #reject, pvals_corrected, alphacSidak, alphacBonf = \
#            #smm.multipletests(p_values.values, method = 'fdr_tsbky')
#            #p_values = pd.Series(data = pvals_corrected, index = p_values.index)
#            #all_pvaluesC[strain][date] = p_values
#            
#            save_name =  strain +'_' + date + '.pdf'
#            plotZStats(z_stats, p_values, z_values, save_name)
#            #print(z_stats['ord_p'])
#            
#            plt.figure(ii_strain)
#            strC = 'brgk'
#            #strC = 'b' if date != 'All' else 'k'
#            plt.plot(np.sort(all_pvalues[strain][date]),'.'+strC[ii_date], label = date)
#            #plt.plot(np.log10(np.sort(all_pvaluesC[strain][date])),'.g')
#            #yy = np.log10(0.05)
#            plt.plot(plt.xlim(), [0.05, 0.05], 'k:')
#            plt.gca().set_yscale('log')
#            plt.title(strain)
#            plt.xlabel('Features')
#            plt.ylabel('p-values')
#            plt.gca().legend(loc=4)
#    
#    for ii_strain, strain in enumerate(strain_list):
#        plt.figure(ii_strain)
#        plt.savefig('p-values_%s.pdf' % strain)
#        plt.close()
#    
#    #%%
#    for strain in strain_list:  
#        strain_pvalues = all_pvalues[strain];
#        aa = (strain_pvalues<0.05).sum();
#        print(strain)
#        print(aa)
#    
#    
#    #%%
#    pd.set_option('display.max_rows', len(feat_list))
#    for strain in strain_list:  
#        strain_pvalues = all_pvalues[strain]
#        max_pvalue = strain_pvalues.drop(['All'], axis=1).max(axis=1);
#        #max_pvalue = (strain_pvalues.drop(['All'], axis=1)<0.05).sum(axis=1)>=2;
#        max_pvalue = max_pvalue[max_pvalue<0.05]
#        max_pvalue.sort()
#        print(strain)
#        print(max_pvalue)
#    pd.reset_option('display.max_rows')
#    
##%%
##import seaborn as sns
# 
##%%
##    pd.set_option('display.max_rows', len(feat_list))
##    for strain in strain_list:  
##        strain_pvalues = all_pvalues[strain]['All']
##        strain_pvalues = strain_pvalues[strain_pvalues<0.05]
##        strain_pvalues.sort()
##        print(strain)
##        print(strain_pvalues)
##    pd.reset_option('display.max_rows')
#
##aa = pvalue_table[strain].sort(inplace=False)   
#
#
##%%
##import matplotlib.pylab as plt
##plt.plot(np.sort(np.log10(pvalue_table['ZR1'].values)))
##plt.plot(np.sort(np.log10(pvalue_table['BR1941'].values)))