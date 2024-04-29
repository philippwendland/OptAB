import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import t, sem
from sklearn.metrics import auc    

import itertools
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(len(s) +1))

with open('/work/wendland//data_prep_three/sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)
    
#3553
with open('/work/wendland//data_prep_three/sepsis_all_1_keys.pkl', 'rb') as handle:
    keys = pickle.load(handle)
key_times=keys

with open('/work/wendland//data_prep_three/sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)

with open('/work/wendland//data_prep_three/sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)

with open('/work/wendland//data_prep_three/sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)

with open('/work/wendland//data_prep_three/sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)

with open('/work/wendland//data_prep_three/sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
    
with open('/work/wendland//data_prep_three/sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)
    
with open('/work/wendland//data_prep_three/sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)


data_active_overall = (~data[:,list(keys.values()),1:2].isnan())
key_times_index = np.array(list(keys.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times_train={list(keys.keys())[x]: keys[x] for x in key_times_index}
key_times=key_times_train

data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test]

data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test]

time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max


with open('/work/wendland//opt_trt_update/best_pred_list_all_corrected.pkl', 'rb') as handle:
    best_pred_list_all = pickle.load(handle)

with open('/work/wendland//opt_trt_update/opt_time_list_all_corrected.pkl', 'rb') as handle:
    opt_time_list_all = pickle.load(handle)

with open('/work/wendland//opt_trt_update/feasible_sol_list_all_corrected.pkl', 'rb') as handle:
    feasible_sol_list_all = pickle.load(handle)
    
with open('/work/wendland//opt_trt_update/feasible_tensors_list_all_corrected.pkl', 'rb') as handle:
    feasible_tensors_list_all = pickle.load(handle)

with open('/work/wendland//opt_trt_update/violated_sol_list_all_corrected.pkl', 'rb') as handle:
    violated_sol_list_all = pickle.load(handle)
    
with open('/work/wendland//opt_trt_update/violated_tensors_list_all_corrected.pkl', 'rb') as handle:
    violated_tensors_list_all = pickle.load(handle)
    
with open('/work/wendland//opt_trt_update/best_treat_int_list_all_corrected.pkl', 'rb') as handle:
    best_treat_int_list_all = pickle.load(handle)


treatment_options_string = list(powerset(['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']))[1:-1]
treatment_options_int = list(powerset([0,1,2]))[1:-1]

diff_to_real_round_minsofa = []
diff_to_real_minsofa = []
diff_to_real_best_trt = []

diff_first = []
diff_first_minsofa = []

for i in range(len(best_pred_list_all)):
    opt_time = opt_time_list_all[i]
    
    #index of last observed value
    nan_index = data_covariables_test[i,:,1].isnan().nonzero(as_tuple=True)[0][0]
    
    #predictions for all treatments
    all_trts = []
    all_trts_fs=[]
    for j in range(len(violated_sol_list_all[i])):
        #k is number of treatment options
        all_trts_j=[]
        all_trts_fs_j=[]
        for k in range(len(treatment_options_int)): 
            if k in violated_sol_list_all[i][j]:
                ind = violated_sol_list_all[i][j].index(k)
                trt = violated_tensors_list_all[i][j][ind]
                all_trts_j.append(trt)
                all_trts_fs_j.append('violated')
            elif k in feasible_sol_list_all[i][j]:
                ind = feasible_sol_list_all[i][j].index(k)
                trt = feasible_tensors_list_all[i][j][ind]
                all_trts_j.append(trt)
                if best_treat_int_list_all[i][j]==treatment_options_int[k]:
                    all_trts_fs_j.append('best')
                else:
                    all_trts_fs_j.append('feasible')
        all_trts.append(all_trts_j)
        all_trts_fs.append(all_trts_fs_j)

    tensor_mins = []
    for j in range(len(all_trts)):
        
        a=torch.stack(all_trts[j])
        b=a[torch.argmin(a[:,0,opt_time[j+1]-opt_time[j]-1,0])]
        b=b[0,:opt_time[j+1]-opt_time[j],0]
        tensor_mins.append(b)
        
    tensor_mins_i = torch.concat(tensor_mins)
    diff_to_real_minsofa_i = tensor_mins_i - (data_X_test[i,opt_time[0]:opt_time[0]+tensor_mins_i.shape[0],0]*variables_std[0]+variables_mean[0])
    diff_to_real_minsofa.append(diff_to_real_minsofa_i.detach())

    a=torch.stack(all_trts[0])
    b=a[torch.argmin(a[:,0,-1,0])]
    b=b[0,:,0]
    first_minsofa=b
    
    diff_first_minsofa_i = first_minsofa - (data_X_test[i,opt_time[0]:opt_time[0]+48,0]*variables_std[0]+variables_mean[0])
    diff_first_minsofa.append(diff_first_minsofa_i.detach())
    
    tensor_best_treat = []
    for j in range(len(best_pred_list_all[i])):
        if best_pred_list_all[i][j] != ['nosep']: 
            b=best_pred_list_all[i][j][:opt_time[j+1]-opt_time[j],0]
            tensor_best_treat.append(b)
    
    tensor_best_treat = torch.concat(tensor_best_treat)
    diff_to_real_best_trt_i = tensor_best_treat - (data_X_test[i,opt_time[0]:opt_time[0]+tensor_best_treat.shape[0],0]*variables_std[0]+variables_mean[0])
    diff_to_real_best_trt.append(diff_to_real_best_trt_i.detach())
    
    diff_first_i = best_pred_list_all[i][0][:,0] - (data_X_test[i,opt_time[0]:opt_time[0]+48,0]*variables_std[0]+variables_mean[0])
    diff_first.append(diff_first_i.detach())
    

# Optimal treatment first timepoint

#Diff
x = torch.empty(size=[711,48])
x[:] = np.nan
for i in range(len(diff_first)):
    x[i,:48] = diff_first[i]

a=np.nanmean(x,axis=0)

x_se = list(sem(x,axis=0,nan_policy='omit'))

quantile = []
n = torch.sum(~x.isnan(),axis=0)
for i in range(x.shape[1]):
    quantile.append(t.ppf(0.975,n[i]))

ci_upper = a + quantile * (x_se/np.sqrt(n.detach().numpy()))
ci_lower = a - quantile * (x_se/np.sqrt(n.detach().numpy()))

ci = quantile * (x_se/np.sqrt(n.detach().numpy()))

auc_x = np.round(auc(x=np.array(range(a.shape[0])),y=a),decimals=2)

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x.shape[1])),a,yerr=ci,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x.shape[1])),a,color='black')

plt.xticks(fontsize=12)
plt.ylim(-1.2,0.1)
plt.yticks(fontsize=12)
plt.title("Area under the curve: " + str(auc_x),fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("Mean $\Delta$ SOFA-Score",fontsize=16,labelpad=-3)

plt.savefig('/work/wendland//GitHub/TE-CDE-main/Bilder_treat_opt/delta_Sofa_diffirst.png')

# Treatment leading to the minimum SOFA first timepoint

x = torch.empty(size=[711,48])
x[:] = np.nan
for i in range(len(diff_first_minsofa)):
    x[i,:48] = diff_first_minsofa[i]

a=np.nanmean(x,axis=0)

x_se = list(sem(x,axis=0,nan_policy='omit'))

quantile = []
n = torch.sum(~x.isnan(),axis=0)
for i in range(x.shape[1]):
    quantile.append(t.ppf(0.975,n[i]))

ci_upper = a + quantile * (x_se/np.sqrt(n.detach().numpy()))
ci_lower = a - quantile * (x_se/np.sqrt(n.detach().numpy()))

ci = quantile * (x_se/np.sqrt(n.detach().numpy()))

auc_x = np.round(auc(x=np.array(range(a.shape[0])),y=a),decimals=2)

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x.shape[1])),a,yerr=ci,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x.shape[1])),a,color='black')

plt.ylim(-1.2,0.1)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Area under the curve: " + str(auc_x),fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("Mean $\Delta$ SOFA-Score",fontsize=16,labelpad=-3)

plt.savefig('/work/wendland//GitHub/TE-CDE-main/Bilder_treat_opt/delta_Sofa_diffirst_minsofa.png')

# Treatment leading to the minimum SOFA over all timepoints

s = 0
for i in diff_to_real_minsofa:
    if i.shape[0]>s:
        s = i.shape[0]

x = torch.empty(size=[711,s])
x[:] = np.nan
for i in range(len(diff_to_real_minsofa)):
    x[i,:diff_to_real_minsofa[i].shape[0]] = diff_to_real_minsofa[i]

x = x[:,torch.sum(~torch.isnan(x),axis=0)>0.05*711]

a=np.nanmean(x,axis=0)

x_se = list(sem(x,axis=0,nan_policy='omit'))

quantile = []
n = torch.sum(~x.isnan(),axis=0)
for i in range(x.shape[1]):
    quantile.append(t.ppf(0.975,n[i]))

ci_upper = a + quantile * (x_se/np.sqrt(n.detach().numpy()))
ci_lower = a - quantile * (x_se/np.sqrt(n.detach().numpy()))

ci = quantile * (x_se/np.sqrt(n.detach().numpy()))

auc_x = np.round(auc(x=np.array(range(a.shape[0])),y=a),decimals=2)

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x.shape[1])),a,yerr=ci,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x.shape[1])),a,color='black')
plt.axvline(48,color='black')
plt.axvline(72,color='black')
plt.axvline(96,color='black')
plt.axvline(120,color='black')

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(-1.2,0.3)
plt.title("Area under the curve: " + str(auc_x),fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("Mean $\Delta$ SOFA-Score",fontsize=16,labelpad=-3)

plt.savefig('/work/wendland//GitHub/TE-CDE-main/Bilder_treat_opt/delta_Sofa_minsofa.png')

# Optimal treatment

s = 0
for i in diff_to_real_best_trt:
    if i.shape[0]>s:
        s = i.shape[0]

x = torch.empty(size=[711,s])
x[:] = np.nan
for i in range(len(diff_to_real_best_trt)):
    x[i,:diff_to_real_best_trt[i].shape[0]] = diff_to_real_best_trt[i]

x = x[:,torch.sum(~torch.isnan(x),axis=0)>0.05*711]

a=np.nanmean(x,axis=0)

x_se = list(sem(x,axis=0,nan_policy='omit'))

quantile = []
n = torch.sum(~x.isnan(),axis=0)
for i in range(x.shape[1]):
    quantile.append(t.ppf(0.975,n[i]))

ci_upper = a + quantile * (x_se/np.sqrt(n.detach().numpy()))
ci_lower = a - quantile * (x_se/np.sqrt(n.detach().numpy()))

ci = quantile * (x_se/np.sqrt(n.detach().numpy()))

auc_x = np.round(auc(x=np.array(range(a.shape[0])),y=a),decimals=2)

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x.shape[1])),a,yerr=ci,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x.shape[1])),a,color='black')
plt.axvline(48,color='black')
plt.axvline(72,color='black')
plt.axvline(96,color='black')
plt.axvline(120,color='black')

plt.xticks(fontsize=12)
plt.ylim(-1.2,0.3)
plt.yticks(fontsize=12)
plt.title("Area under the curve: " + str(auc_x),fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("Mean $\Delta$ SOFA-Score",fontsize=16,labelpad=-3)

plt.savefig('/work/wendland//GitHub/TE-CDE-main/Bilder_treat_opt/delta_Sofa_besttrt.png')


