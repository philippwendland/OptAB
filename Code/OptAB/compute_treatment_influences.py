import torch
import numpy as np
import utils_paper
import pandas as pd
import itertools

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

with open('/work/wendland/opt_time_list_all.pkl', 'rb') as handle:
    opt_time_list_all = pickle.load(handle)
# List including all treatment optimization timepoints. Used in this context to determine treatment optimization start for plots
    
# Possible to do this only on covariables

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
data_treat_thresh = None

# Initializing and loading the Encoder
model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables))
model=model.to(model.device)
model.load_state_dict(torch.load('/work/wendland/Trained_Encoder.pth'))

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

### Starting with predictions for Vancomycin

# Initializing list including predictions
predictions_list = []
pred_horizon = 48
for k in range(data_X_test.shape[0]):
    print('patient number' + str(k))
    
    # Creating subsets of tensors (3 indices due to torchcde implementation)
    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    
    # Treatment start
    opt_time = opt_time_list_all[k]
    start_pred = opt_time[0]
    
    # Setting treatments to Vancomycin
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=1
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=0
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=0
    data_treatment_update_k2[:,start_pred:,0] = 1
    data_treatment_update_k2[:,start_pred:,1] = 0
    data_treatment_update_k2[:,start_pred:,2] = 0
    
    # Computing predictions
    pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
    
    # First index
    pred_X_val=pred_X_val[0:1,:,:]
    
    # De-normalize predictions
    pred_X_val_unsc = pred_X_val[:,:,:].clone()
    pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
    pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    predictions_list.append(pred_X_val_unsc[0])

with open('/work/wendland/opt_trt_update/tr_pred_vanco.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

### PipTaz

# Initializing list including predictions
predictions_list = []
pred_horizon = 48
for k in range(data_X_test.shape[0]):
    print('patient number' + str(k))
    
    # Creating subsets of tensors (3 indices due to torchcde implementation)
    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    
    # Treatment start
    opt_time = opt_time_list_all[k]
    start_pred = opt_time[0]
    
    # Setting treatments to Vancomycin
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=0
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=1
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=0
    data_treatment_update_k2[:,start_pred:,0] = 0
    data_treatment_update_k2[:,start_pred:,1] = 1
    data_treatment_update_k2[:,start_pred:,2] = 0
    
    # Computing predictions
    pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
    
    # First index
    pred_X_val=pred_X_val[0:1,:,:]
    
    # De-normalize predictions
    pred_X_val_unsc = pred_X_val[:,:,:].clone()
    pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
    pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    predictions_list.append(pred_X_val_unsc[0])

with open('/work/wendland/opt_trt_update/tr_pred_piptaz.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
### Ceftriaxone

# Initializing list including predictions
predictions_list = []
pred_horizon = 48
for k in range(data_X_test.shape[0]):
    print('patient number' + str(k))
    
    # Creating subsets of tensors (3 indices due to torchcde implementation)
    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    
    # Treatment start
    opt_time = opt_time_list_all[k]
    start_pred = opt_time[0]
    
    # Setting treatments to Vancomycin
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=0
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=0
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=1
    data_treatment_update_k2[:,start_pred:,0] = 0
    data_treatment_update_k2[:,start_pred:,1] = 0
    data_treatment_update_k2[:,start_pred:,2] = 1
    
    # Computing predictions
    pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
    
    # First index
    pred_X_val=pred_X_val[0:1,:,:]
    
    # De-normalize predictions
    pred_X_val_unsc = pred_X_val[:,:,:].clone()
    pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
    pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    predictions_list.append(pred_X_val_unsc[0])

with open('/work/wendland/opt_trt_update/tr_pred_ceftri.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

### Vancomycin + Piperacillin/Tazobactam

# Initializing list including predictions
predictions_list = []
pred_horizon = 48
for k in range(data_X_test.shape[0]):
    print('patient number' + str(k))
    
    # Creating subsets of tensors (3 indices due to torchcde implementation)
    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    
    # Treatment start
    opt_time = opt_time_list_all[k]
    start_pred = opt_time[0]
    
    # Setting treatments to Vancomycin
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=1
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=1
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=0
    data_treatment_update_k2[:,start_pred:,0] = 1
    data_treatment_update_k2[:,start_pred:,1] = 1
    data_treatment_update_k2[:,start_pred:,2] = 0
    
    # Computing predictions
    pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
    
    # First index
    pred_X_val=pred_X_val[0:1,:,:]
    
    # De-normalize predictions
    pred_X_val_unsc = pred_X_val[:,:,:].clone()
    pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
    pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    predictions_list.append(pred_X_val_unsc[0])

with open('/work/wendland/opt_trt_update/tr_pred_vancopiptaz.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

### Piperacillin/Tazobactam + Ceftriaxone

# Initializing list including predictions
predictions_list = []
pred_horizon = 48
for k in range(data_X_test.shape[0]):
    print('patient number' + str(k))
    
    # Creating subsets of tensors (3 indices due to torchcde implementation)
    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    
    # Treatment start
    opt_time = opt_time_list_all[k]
    start_pred = opt_time[0]
    
    # Setting treatments to Vancomycin
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=0
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=1
    data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=1
    data_treatment_update_k2[:,start_pred:,0] = 0
    data_treatment_update_k2[:,start_pred:,1] = 1
    data_treatment_update_k2[:,start_pred:,2] = 1
    
    # Computing predictions
    pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
    
    # First index
    pred_X_val=pred_X_val[0:1,:,:]
    
    # De-normalize predictions
    pred_X_val_unsc = pred_X_val[:,:,:].clone()
    pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
    pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    
    predictions_list.append(pred_X_val_unsc[0])

with open('/work/wendland/opt_trt_update/tr_pred_piptazcef.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(len(s) +1))


treatment_options_string = list(powerset(['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']))[1:-1]
treatment_options_int = list(powerset([0,1,2]))[1:-1]

predictions_list = []
p=False
for k in range(data_X_test.shape[0]):
    
    print('patient number' + str(k))
    
    predictions_list_k=[]

    data_X_test_k = data_X_test[k:k+3,:,:].clone()
    data_toxic_test_k = data_toxic_test[k:k+3,:,:].clone()
    data_time_test_k = data_time_test[k:k+3,:,:].clone()
    data_active_test_k = data_active_test[k:k+3,:,:].clone()
    data_static_test_k = data_static_test[k:k+3,:].clone()
        
    
    data_covariables_update_k2 = data_covariables_test[k:k+3,:,:].clone()
    data_treatment_update_k2 = data_treatment_test[k:k+3,:,:].clone()
    

    opt_time = opt_time_list_all[k]
    
    #for i in range(len(opt_time)-1):
    
    start_pred = opt_time[0]
    pred_horizon = 48

    pred_list_k = []
    while ~data_X_test[k,start_pred].isnan():
        
        pred_list_k_iteration=[]
        
        for i in range(len(treatment_options_string)):                
            treat_opt_str = treatment_options_string[i]
            treat_opt_int = treatment_options_int[i]
            
            data_covariables_update_k2[:,start_pred:,variables_complete.index('Vancomycin')]=0
            data_covariables_update_k2[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=0
            data_covariables_update_k2[:,start_pred:,variables_complete.index('Ceftriaxon')]=0
            data_treatment_update_k2[:,start_pred:,0] = 0
            data_treatment_update_k2[:,start_pred:,1] = 0
            data_treatment_update_k2[:,start_pred:,2] = 0
            
            for j in range(len(treat_opt_str)):
                data_covariables_update_k2[:,start_pred:,variables_complete.index(treat_opt_str[j])]=1
                data_treatment_update_k2[:,start_pred:,treat_opt_int[j]] = 1
            
            pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k2, covariables=data_covariables_update_k2, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)      
            
            pred_X_val=pred_X_val[0:1,:,:] #just taking first sample index
            
            pred_X_val_unc = pred_X_val[:,:,:].clone()
            pred_X_val_unc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
            pred_X_val_unc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
            pred_X_val_unc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
            pred_X_val_unc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
            
            pred_list_k_iteration.append(pred_X_val_unc)
        pred_k_iteration = torch.stack(pred_list_k_iteration)
        min_sofa_ind = torch.argmin(pred_k_iteration[:,0,-1,0])
        pred_list_k.append(pred_k_iteration[min_sofa_ind])
        
        pred_horizon=24
        start_pred=start_pred+pred_horizon 
           
    pred_k = torch.concat(pred_list_k,axis=1)[0]
            
    predictions_list.append(pred_k)

with open('/work/wendland/opt_trt_update/tr_minsofa_horizon.pkl', 'wb') as handle:
    pickle.dump(predictions_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
