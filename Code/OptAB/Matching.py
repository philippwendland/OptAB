import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import t, sem

import itertools
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(len(s) +1))

data = pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_lab.pkl')

keys=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_keys.pkl')
key_times=keys

variables_complete=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_variables_complete.pkl')

static_tensor=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_static.pkl')

static_variables=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_static_variables.pkl')

variables_mean=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_variables_mean.pkl')

variables_std=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_variables_std.pkl')
    
indices_test=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_indices_test.pkl')
    
variables=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_variables.pkl')

creat_dat=pd.read_pickle('/work/wendland/opt_treat/creat_dat.pkl')

microbiol_dat=pd.read_pickle('/work/wendland/opt_treat/microbiol_dat_.pkl')

static_mean=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_static_mean.pkl')
    
static_std=pd.read_pickle('/work/wendland/opt_treat/sepsis_all_1_static_std.pkl')

microbiol_fungal = pd.read_pickle('/work/wendland/opt_treat/microbiol_fungal_dat_.pkl')

with open('/work/wendland/opt_trt_update/best_pred_list_all_corrected.pkl', 'rb') as handle:
    best_pred_list_all = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/best_treat_list_all_corrected.pkl', 'rb') as handle:
    best_treat_list_all = pickle.load(handle)

with open('/work/wendland/opt_trt_update/opt_time_list_all_corrected.pkl', 'rb') as handle:
    opt_time_list_all = pickle.load(handle)

with open('/work/wendland/opt_trt_update/feasible_sol_list_all_corrected.pkl', 'rb') as handle:
    feasible_sol_list_all = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/feasible_tensors_list_all_corrected.pkl', 'rb') as handle:
    feasible_tensors_list_all = pickle.load(handle)

with open('/work/wendland/opt_trt_update/violated_sol_list_all_corrected.pkl', 'rb') as handle:
    violated_sol_list_all = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/violated_tensors_list_all_corrected.pkl', 'rb') as handle:
    violated_tensors_list_all = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/best_treat_int_list_all_corrected.pkl', 'rb') as handle:
    best_treat_int_list_all = pickle.load(handle)


## "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_test])
key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times_train={list(key_times.keys())[x]: key_times[x] for x in key_times_index}
key_times=key_times_train

#Extracting outcome and side-effects from data and setting device
data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test]
data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic=data_toxic[:,:,None][indices_test]
data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2=data_toxic2[:,:,None][indices_test]
data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3=data_toxic3[:,:,None][indices_test]
data_toxic_test = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)

#Extracting treatments and side-effects from data and setting device
data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment=data_treatment[:,:,None][indices_test]
data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2=data_treatment2[:,:,None][indices_test]
data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3=data_treatment3[:,:,None][indices_test]
data_treatment_test = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)

#Extracting the covariables
data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test]

#Normalizing the missing masks to one
time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max

# Selection of training and test data
data_time_test = data[:,:list(key_times.keys())[-1],0:1][indices_test]
data_active_test = ~data[:,list(key_times.values()),1:2].isnan()[indices_test]
data_static_test=static_tensor[indices_test]

treatment_options_string = list(powerset(['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']))[1:-1]
treatment_options_int = list(powerset([0,1,2]))[1:-1]

trt_string = ['Vancomycin','Pip/Taz','Ceftriaxone','Vanco, Pip/Taz','Vanco, Cef','Pip/Taz, Cef']

# Focus on first treatment iteration

# Vector with the initialized treatments
start_treats = torch.zeros(len(opt_time_list_all),3)

for i in range(len(opt_time_list_all)):
    start_treats[i]=data_treatment_test[i,opt_time_list_all[i][0],:]

# list with the indices of the patient with the smallest distance
list_nearest_indices = []
# list with the distances to the patient with the smallest distance
list_nearest_distances = []

# Tensor with the covariables at treatment initialization
covariables_treat_int = torch.zeros((len(best_treat_int_list_all)),11)
covariables_treat_int[:] = np.nan
for i in range(len(best_treat_int_list_all)):
    covariables_treat_int[i] = data_covariables_test[i,opt_time_list_all[i][0],1:12]

# "Counterfactual Matching"
# Iterating over all patients
for i in range(len(best_treat_int_list_all)):
    
    patient_i_opt_treat = best_treat_int_list_all[i][0]
    
    mask = torch.ones(3,dtype=torch.bool)
    mask[list(patient_i_opt_treat)] = False
    
    # List of patients for whom the factual treatment corresponds to OPtAB's proposed optimal treatment for patient i
    patients_with_best_treat_ind=[]
    
    if not( (start_treats[i,patient_i_opt_treat]==1).all() and (start_treats[i,mask]==0).all()):
        
        # checking if the factual treatment corresponds to OPtAB's proposed optimal treatment for patient i
        for j in range(len(best_treat_int_list_all)):
            if (start_treats[j,patient_i_opt_treat]==1).all() and (start_treats[j,mask]==0).all() and opt_time_list_all[j][0]<3:
                patients_with_best_treat_ind.append(j)
        
        # Focus on covariables, not missing masks
        patient_i_covariables = data_covariables_test[i,0,1:12]

        patient_compare_covariables = data_covariables_test[patients_with_best_treat_ind,0,1:12]

        distances = np.nanmean((patient_compare_covariables-patient_i_covariables)**2,axis=1)
        
        # Exclude patients, for whom treatment initialization was later than 3 hours after Sepsis onset
        if len(distances)>0 and opt_time_list_all[i][0]<3:
            nearest_index = patients_with_best_treat_ind[np.argmin(distances)]
            nearest_distance = np.min(distances)
        else:
            nearest_index = np.nan
            nearest_distance = np.nan
    
    else:
        nearest_index=np.nan
        nearest_distance = np.nan
    
    list_nearest_indices.append(nearest_index)
    list_nearest_distances.append(nearest_distance)
    
i=0
mae = np.zeros((len(list_nearest_indices),48),)
mae[:] = np.nan

for i in range(len(list_nearest_indices)):
    if pd.notna(list_nearest_indices[i]):# and list_nearest_distances[i]<1:
        k=np.abs(best_pred_list_all[i][0][:,0].detach().numpy()-((data_X_test[list_nearest_indices[i],opt_time_list_all[list_nearest_indices[i]][0]:opt_time_list_all[list_nearest_indices[i]][0]+48,0])*variables_std[0]+variables_mean[0]).detach().numpy())    
        mae[i] =k
        
x=mae        
a=np.nanmean(x,axis=0)
        
x_se = list(sem(x,axis=0,nan_policy='omit'))

quantile = []
n = np.sum(pd.notna(x),axis=0)
for i in range(x.shape[1]):
    quantile.append(t.ppf(0.975,n[i]))

ci_upper = a + quantile * (x_se/np.sqrt(n))
ci_lower = a - quantile * (x_se/np.sqrt(n))

ci = quantile * (x_se/np.sqrt(n))

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x.shape[1])),a,yerr=ci,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x.shape[1])),a,color='black')

plt.ylim(0,3.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Counterfactual Matching",fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("MAE | matched SOFA-Scores |",fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/matched_SOFA_Score.png')



        
list_nearest_indices_sametrt = []
list_nearest_distances_sametrt = []

covariables_treat_int = torch.zeros((len(best_treat_int_list_all)),11)
covariables_treat_int[:] = np.nan

# "Factual Matching"
for i in range(len(best_treat_int_list_all)):
    
    patient_i_start_treat = start_treats[i]
    
    patients_with_same_treat_ind=[]
    
    # Mapping to patients with the same treatment
    for j in range(len(best_treat_int_list_all)):
        if (start_treats[j]==patient_i_start_treat).all() and opt_time_list_all[j][0]<3 and j != i: #(start_treats[i,mask]==0).all())
            patients_with_same_treat_ind.append(j)    
    patient_i_covariables = data_covariables_test[i,0,1:12]

    patient_compare_covariables = data_covariables_test[patients_with_same_treat_ind,0,1:12]
    
    distances = np.nanmean((patient_compare_covariables-patient_i_covariables)**2,axis=1)
    
    if len(distances)>0 and opt_time_list_all[i][0]<3:
        nearest_index = patients_with_same_treat_ind[np.argmin(distances)]
        nearest_distance = np.min(distances)
    else:
        nearest_index = np.nan
        nearest_distance = np.nan
    
    list_nearest_indices_sametrt.append(nearest_index)
    list_nearest_distances_sametrt.append(nearest_distance)
    
i=0
mae_fac = np.zeros((len(list_nearest_indices),48),)
mae_fac[:] = np.nan

for i in range(len(list_nearest_indices_sametrt)):
    if pd.notna(list_nearest_indices_sametrt[i]):# and list_nearest_distances[i]<1:
        k=np.abs(((data_X_test[i,opt_time_list_all[i][0]:opt_time_list_all[i][0]+48,0])*variables_std[0]+variables_mean[0]).detach().numpy()-((data_X_test[list_nearest_indices_sametrt[i],opt_time_list_all[list_nearest_indices_sametrt[i]][0]:opt_time_list_all[list_nearest_indices_sametrt[i]][0]+48,0])*variables_std[0]+variables_mean[0]).detach().numpy())
        
        mae_fac[i] =k
        
x_fac=mae_fac        
a_fac=np.nanmean(x_fac,axis=0)
        
x_se_fac = list(sem(x_fac,axis=0,nan_policy='omit'))

quantile_fac = []
n_fac = np.sum(pd.notna(x_fac),axis=0)
for i in range(x_fac.shape[1]):
    quantile_fac.append(t.ppf(0.975,n[i]))

ci_upper_fac = a_fac + quantile_fac * (x_se_fac/np.sqrt(n_fac))
ci_lower_fac = a_fac - quantile_fac * (x_se_fac/np.sqrt(n_fac))

ci_fac = quantile_fac * (x_se_fac/np.sqrt(n_fac))

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.errorbar(list(range(x_fac.shape[1])),a_fac,yerr=ci_fac,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')
plt.plot(list(range(x_fac.shape[1])),a_fac,color='black')

plt.ylim(0,3.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Factual Matching",fontsize=16)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("MAE | matched SOFA-Scores |",fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/matched_SOFA_Score_factual.png')

fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)
plt.plot(list(range(x.shape[1])),a,color='black',label='Counterfactual Matching')
plt.errorbar(list(range(x.shape[1])),a,yerr=ci_fac,color='red',linestyle='',capsize=2,capthick=1)

plt.plot(list(range(x_fac.shape[1])),a_fac,color='dodgerblue',label='Factual Matching')
plt.errorbar(list(range(x_fac.shape[1])),a_fac,yerr=ci_fac,color='red',linestyle='',capsize=2,capthick=1,label='95 % CI')

plt.ylim(0,3.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Time in hours",fontsize=16)
plt.legend()
plt.ylabel("MAE | matched SOFA-Scores |",fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/matched_SOFA_Score_both.png')
