import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.stats import gaussian_kde

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)
# Tensor of patient data, size: number of patients x timepoints x variables (including one dimension corresponding to time)

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_keys.pkl', 'rb') as handle:
    key_times = pickle.load(handle)
# Dictionary where keys correspond to the index and the values correspond to the time in hours

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)
# List of all variable names of the time-dependent variables

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)
# Tensor of static variables, size: number of patients x static_variables

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)
# List of all static variable names

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)
# list of means of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)

with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
# list of standard deviations of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)
    
with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)
    
with open('/work/wendland/tecde_three_compdepth/pictures/sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)
# list of index variables for the variables to be standardized

# List of treatment predictions
with open('/work/wendland/opt_trt_update/tr_pred_ceftri.pkl', 'rb') as handle:
    trt_start_ceftri = pickle.load(handle)
        
with open('/work/wendland/opt_trt_update/tr_pred_piptaz.pkl', 'rb') as handle:
    trt_start_piptaz = pickle.load(handle)
        
with open('/work/wendland/opt_trt_update/tr_pred_vanco.pkl', 'rb') as handle:
    trt_start_vanco = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/tr_pred_vancopiptaz.pkl', 'rb') as handle:
    trt_start_vancopiptaz = pickle.load(handle)
    
with open('/work/wendland/opt_trt_update/tr_pred_piptazcef.pkl', 'rb') as handle:
    trt_start_piptazcef = pickle.load(handle)
        

# Plot and color labels
labels = ['Vancomycin','Piperacillin-Tazobactam','Ceftriaxone']
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Creating tensors of the predictions for the different treatments
# size: number of patients x timepoints (here 100) x number of variables
trt_start_vanco = torch.stack(trt_start_vanco)
trt_start_ceftri = torch.stack(trt_start_ceftri)
trt_start_piptaz = torch.stack(trt_start_piptaz)
trt_start_vancopiptaz = torch.stack(trt_start_vancopiptaz)
trt_start_piptazcef = torch.stack(trt_start_piptazcef)


if torch.cuda.is_available():
    device_type = "cuda"
else:
    device_type = "cpu"
    
device = torch.device(device_type)

## "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_test])
key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times_train={list(key_times.keys())[x]: key_times[x] for x in key_times_index}
key_times=key_times_train

#Extracting outcome and side-effects from data and setting device
data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test].to(device)
data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic=data_toxic[:,:,None][indices_test].to(device)
data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2=data_toxic2[:,:,None][indices_test].to(device)
data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3=data_toxic3[:,:,None][indices_test].to(device)
data_toxic_test = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)

#Extracting treatments and side-effects from data and setting device
data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment=data_treatment[:,:,None][indices_test].to(device)
data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2=data_treatment2[:,:,None][indices_test].to(device)
data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3=data_treatment3[:,:,None][indices_test].to(device)
data_treatment_test = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)


vanc_ind = torch.nonzero(torch.nansum(data_treatment_test[:,:,0]==1,axis=1)>0).squeeze().tolist() #409 Vancomycin patients
ceft_ind = torch.nonzero(torch.nansum(data_treatment_test[:,:,2]==1,axis=1)>0).squeeze().tolist()#298 Ceftriaxon patients
piptaz_ind = torch.nonzero(torch.nansum(data_treatment_test[:,:,1]==1,axis=1)>0).squeeze().tolist()#232 Ceftriaxon patients

vanc_ceft_ind = torch.nonzero(torch.logical_and(torch.nansum(data_treatment_test[:,:,0]==1,axis=1)>0,torch.nansum(data_treatment_test[:,:,2]==1,axis=1)>0)) #54
piptaz_ceft_ind = torch.nonzero(torch.logical_and(torch.nansum(data_treatment_test[:,:,1]==1,axis=1)>0,torch.nansum(data_treatment_test[:,:,2]==1,axis=1)>0)) #18
vanc_piptaz_ind = torch.nonzero(torch.logical_and(torch.nansum(data_treatment_test[:,:,0]==1,axis=1)>0,torch.nansum(data_treatment_test[:,:,1]==1,axis=1)>0)) #171

vanc_ind = [x for x in vanc_ind if (x not in vanc_ceft_ind and x not in piptaz_ceft_ind and x not in vanc_piptaz_ind)] #199
ceft_ind = [x for x in ceft_ind if (x not in vanc_ceft_ind and x not in piptaz_ceft_ind and x not in vanc_piptaz_ind)] #241
piptaz_ind = [x for x in piptaz_ind if (x not in vanc_ceft_ind and x not in piptaz_ceft_ind and x not in vanc_piptaz_ind)] #58

seed=21786

np.random.seed(seed)
    
sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]    

month_names = np.arange(1,49)
month_counts = trt_start_ceftri2[:,:48,2]-trt_start_vanco2[:,:48,2]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()


max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-1,2, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)

for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Bilirubin total',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)

for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=100000000)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Cef-Van_Bil_48.png')


seed = 530

np.random.seed(seed)

sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]        


month_names = np.arange(1,49)
month_counts = trt_start_ceftri2[:,:48,3]-trt_start_vanco2[:,:48,3]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()


max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-100,100, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)
for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Alanine Transaminase',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)
for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=10000000000)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Cef-Van_Alt_48.png')

seed = 7109

np.random.seed(seed)

sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]        

month_names = np.arange(1,49)
month_counts = trt_start_ceftri2[:,:48,1]-trt_start_vanco2[:,:48,1]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()

max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-0.3,0.3, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)
for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Creatinine',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)
for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=10000000000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Van-Cef_Crea_48.png')


seed=21786

np.random.seed(seed)
    
sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]    

month_names = np.arange(1,73)
month_counts = trt_start_ceftri2[:,:72,2]-trt_start_vanco2[:,:72,2]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()

max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-1.5,2.2, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)

for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Bilirubin total',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)
for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=100000000)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Cef-Van_Bil_72.png')

seed = 530

np.random.seed(seed)

sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]        

month_names = np.arange(1,73)
month_counts = trt_start_ceftri2[:,:72,3]-trt_start_vanco2[:,:72,3]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()

max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-100,100, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)
for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Alanine Transaminase',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)
for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=10000000000)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Cef-Van_Alt_72.png')

seed = 7109

np.random.seed(seed)

sampled_indices = np.random.choice(vanc_ind,size=58,replace=False)
ceft_ind2 = np.random.choice(ceft_ind,size=58,replace=False)
sampled_indices = list(sampled_indices) + list(ceft_ind2) + list(piptaz_ind)
sampled_indices = list(set(sampled_indices))
    
trt_start_ceftri2 = trt_start_ceftri[sampled_indices]
trt_start_vanco2 = trt_start_vanco[sampled_indices]        

month_names = np.arange(1,73)
month_counts = trt_start_ceftri2[:,:72,1]-trt_start_vanco2[:,:72,1]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()

med = np.nanmedian(month_counts,axis=1)
mean = np.nanmean(month_counts,axis=1)
quart_1 = np.nanquantile(month_counts,q=0.25,axis=1)
quart_2 = np.nanquantile(month_counts,q=0.75,axis=1)
quant_1 = np.nanquantile(month_counts,q=0.10,axis=1)
quant_2 = np.nanquantile(month_counts,q=0.9,axis=1)

max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-0.3,0.3, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)
for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Creatinine',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)

for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=10000000000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Van-Cef_Crea_72.png')

month_names = np.arange(1,73)
month_counts = trt_start_ceftri2[:,:72,1]-trt_start_vanco2[:,:72,1]
month_counts = torch.swapaxes(month_counts.detach(),axis0=1,axis1=0).detach()


max_count = max([count_i.max() for count_i in month_counts])
min_count = min([count_i.min() for count_i in month_counts])

xs = np.linspace(-0.3,0.3, 200)
month_kde = [gaussian_kde(count_i, bw_method=0.2) for count_i in month_counts]
max_kde = max([kde_i(xs).max() for kde_i in month_kde])
overlap_factor = 1.9

fig, ax = plt.subplots(figsize=(5.5,5.5),dpi=600)
for index in range(1,len(month_names),2):
    kde = month_kde[index](xs) / max_kde * overlap_factor
    ax.plot(xs, index + kde, lw=2, color='black', zorder=50 - 2 * index + 1)
    fill_poly = ax.fill_between(xs, index, index + kde, color='none', alpha=0.8)

    verts = np.vstack([p.vertices for p in fill_poly.get_paths()])
    gradient = ax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='magma', aspect='auto', zorder=50 - 2 * index,
                         extent=[verts[:, 0].min(), verts[:, 0].max(), verts[:, 1].min(), verts[:, 1].max()])
    gradient.set_clip_path(fill_poly.get_paths()[0], transform=plt.gca().transData)

ax.set_xlim(xs[0], xs[-1])
ax.set_ylim(ymin=-0.2)

ax.set_xlabel('$\Delta$ Creatinine',fontsize=16)
ax.set_ylabel('Time in hours',fontsize=16)
ax.set_title('Ceftriaxone - Vancomycin',fontsize=16)
for spine in ('top', 'left', 'right'):
    ax.spines[spine].set(visible=False)
plt.axvline(0,color='black',zorder=10000000000)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.legend(loc='upper right').set_zorder(100000000000)

plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_new_ITE/Van-Cef_Crea_72_mean.png')


