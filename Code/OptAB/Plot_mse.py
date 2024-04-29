import torch
import numpy as np
import matplotlib.pyplot as plt
import utils_paper
import pandas as pd
import pickle            

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)
# Tensor of patient data, size: number of patients x timepoints x variables (including one dimension corresponding to time)

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_keys.pkl', 'rb') as handle:
    key_times = pickle.load(handle)
# Dictionary where keys correspond to the index and the values correspond to the time in hours

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)
# List of all variable names of the time-dependent variables

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)
# Tensor of static variables, size: number of patients x static_variables

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)
# List of all static variable names

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)
# list of means of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)

with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
# list of standard deviations of the variables to be standardized (not all variables should be standardized due to missing masks or the time channel, key is the variables)
    
with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)
    
with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)
# list of index variables for the variables to be standardized

# hyperparameters of the Encoder
hidden_channels = 17
#batch_size = 500
hidden_states = 33
#lr = 0.0050688746606452565
activation = 'tanh'
num_depth = 15
pred_act = 'tanh'
pred_states = 128
pred_depth = 1
pred_comp=True


# Threshold for the model to compute only positive outputs (via softplus)
data_thresh = ((0-variables_mean)/variables_std)[[variables.index("SOFA"),variables.index("creatinine"),variables.index("bilirubin_total"),variables.index("alt")]]

# Initializing and loading the Encoder
model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables))
model=model.to(model.device)
model.load_state_dict(torch.load('/work/wendland/Trained_Encoder.pth',map_location=torch.device('cpu')))


# Hyperparameters of Decoder
hidden_channels_dec = 17
batch_size_dec = 1000
hidden_states_dec = 33
lr_dec = 0.0050688746606452565
activation_dec = 'tanh'
num_depth_dec = 15
pred_act_dec = 'tanh'
pred_states_dec = 128
pred_depth_dec = 1

offset=0

# determined by data
output_channels=4 # Number of (all) outcomes 
input_channels_dec=1 # Only time as control
z0_hidden_dimension_dec = hidden_channels + 1 + static_tensor.shape[-1] + 6

# Initializing Decoder
model_decoder = utils_paper.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=z0_hidden_dimension_dec,activation=activation_dec,num_depth=num_depth_dec, pos=True, thresh=data_thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec, treatment_options=3)
model_decoder=model_decoder.to(model_decoder.device)
model_decoder.load_state_dict(torch.load('/work/wendland/Trained_Decoder.pth',map_location=torch.device('cpu')))


rectilinear_index=0

#### "Pre-Processing of data"

## "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_test])
key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
key_times_train={list(key_times.keys())[x]: key_times[x] for x in key_times_index}
key_times=key_times_train

#Extracting outcome and side-effects from data and setting device
data_X_test = data[:,list(key_times.values())[1:],1:2][indices_test].to(model.device)
data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
data_toxic=data_toxic[:,:,None][indices_test].to(model.device)
data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
data_toxic2=data_toxic2[:,:,None][indices_test].to(model.device)
data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
data_toxic3=data_toxic3[:,:,None][indices_test].to(model.device)
data_toxic_test = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)

#Extracting treatments and side-effects from data and setting device
data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
data_treatment=data_treatment[:,:,None][indices_test].to(model.device)
data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
data_treatment2=data_treatment2[:,:,None][indices_test].to(model.device)
data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
data_treatment3=data_treatment3[:,:,None][indices_test].to(model.device)
data_treatment_test = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)

#Extracting the covariables
data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test].to(model.device)

#Normalizing the missing masks to one
time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max

# Selection of training and test data
data_time_test = data[:,:list(key_times.keys())[-1],0:1][indices_test].to(model.device)
data_active_test = ~data[:,list(key_times.values()),1:2].isnan()[indices_test].to(model.device)
data_static_test=static_tensor[indices_test].to(model.device,dtype=torch.float32)    


# Compute unscaled data
data_toxic_test_unscaled=data_toxic_test.clone()
data_toxic_test_unscaled[:,:,0] = data_toxic_test_unscaled[:,:,0]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
data_toxic_test_unscaled[:,:,1] = data_toxic_test_unscaled[:,:,1]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
data_toxic_test_unscaled[:,:,2] = data_toxic_test_unscaled[:,:,2]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]

unscaled=False

data_X_test =data_X_test[:10]
data_toxic_test=data_toxic_test[:10]
data_treatment_test=data_treatment_test[:10]
data_covariables_test = data_covariables_test[:10]
data_time_test = data_time_test[:10]
data_active_test = data_active_test[:10]
data_static_test = data_static_test[:10]

# import matplotlib
# matplotlib.rcParams.update({'font.size': 5})
# fig, ax = plt.subplots(figsize=(1,4),dpi=400)
# heatmap=ax.pcolor([[0,1]],cmap=plt.cm.seismic)
# ax.set_visible(False)
# fig.colorbar(heatmap)
# plt.savefig('C:/Users/wendland/Documents/GitHub/TE-CDE-main/Bilder_dec_three_nomed_encbatch_batch_3_atestplots/colorbar.png')

#Normal plots
offset=0
max_horizon=72
title='SOFA-Score '
step=None
save_link = None
load_map = None
index=0
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='Creatinine '
step=5
save_link = None
load_map = None
index=1
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='Bilirubin '
step=5
save_link = None
load_map = None
index=2
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='ALT '
step=5
save_link = None
load_map = None
index=3
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)


#Plots without vmax
offset=0
max_horizon=72
title='SOFA-Score '
step=None
save_link = None
load_map = None
index=0
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='Creatinine '
step=5
save_link = None
load_map = None
index=1
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='Bilirubin '
step=5
save_link = None
load_map = None
index=2
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=72
title='ALT '
step=5
save_link = None
load_map = None
index=3
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='SOFA-Score '
step=None
save_link = None
load_map = None
index=0
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='Creatinine '
step=5
save_link = None
load_map = None
index=1
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='Bilirubin '
step=5
save_link = None
load_map = None
index=2
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='ALT '
step=5
save_link = None
load_map = None
index=3
vmin=0
vmax=1
colorbar=False
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)


#Plots without vmax
offset=0
max_horizon=96
title='SOFA-Score '
step=None
save_link = None
load_map = None
index=0
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='Creatinine '
step=5
save_link = None
load_map = None
index=1
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='Bilirubin '
step=5
save_link = None
load_map = None
index=2
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)

offset=0
max_horizon=96
title='ALT '
step=5
save_link = None
load_map = None
index=3
vmin=0
vmax=None
colorbar=True
utils_paper.heatmap_pred_dec(model, model_decoder, offset=offset, max_horizon=max_horizon,loss='mse', unscaled=unscaled, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables=data_covariables_test, time_covariates=data_time_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index, step=step, variables_std=variables_std,variables_mean=variables_mean,variables=variables, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True,save_link=save_link,load_map=load_map,title=title,vmin=vmin,vmax=vmax,index=index,colorbar=colorbar)



