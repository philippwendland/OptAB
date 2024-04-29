import math
import torch
import numpy as np
import utils_paper
import pandas as pd
import pickle

import itertools

# Function to create sets based on lists
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


creat_dat_test = [creat_dat[i] for i in indices_test]
microbiol_dat_test = [microbiol_dat[i] for i in indices_test]

# List of all Antibiotic combinations
treatment_options_string = list(powerset(['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']))[1:-1]
treatment_options_int = list(powerset([0,1,2]))[1:-1]

# Side-effects associated thresholds
sideeffects_m = [2, 2.4, 280]
sideeffects_f = [2, 2.2, 280]

# Threshold for including measured microbiological data (before start (!))
microbiol_time=168

# Starting timepoint
start_timepoint='first_treatment'


feasible_sol_list_all=[] # 
# Nested list including indices for all feasible solutions of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations (number of feasible solutions))

sofa_min_arg_list_all=[] 
# Nested list including indices for the predicted optimal treatment leading to minimal SOFA-Score of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations)

best_treat_list_all=[] 
# Nested list including tuple of antibiotic strings of the predicted optimal treatment leading to minimal SOFA-Score of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations)

sofa_min_list_all=[] 
# Nested list including the predicted minimal SOFA-Score of the optimal treatment of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations)

feasible_tensors_list_all=[] 
# Nested list including predictions of all feasible solutions of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations (Torch Tensor, size: number feasible solutions x 1 x timepoints until next iteration x number of outcomes))

best_treat_int_list_all=[]
# Nested list including tuple of indices of the treatment leading to minimal SOFA-Score of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations)

opt_time_list_all=[]
# Nested list including all timepoints of the start and end of optimal treatment selection, first tp is only start and last tp only end
# size: number of patients (number of patient specific treatment optimizations + 1)

best_pred_list_all=[]
# Nested list including predictions of the optimal treatment of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations (Torch Tensor, size: timepoints until next iteration x number of outcomes))


violated_sol_list_all=[]
# Nested list including indices for all solutions violating side-effects associated thresholds of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations (number of solutions))

violated_tensors_list_all=[]
# Nested list including predictions of all solutions violating side-effects associated thresholds of all treatment optimizations of all patients 
# size: number of patients (number of patient specific treatment optimizations (Torch Tensor, size: number of solutions x 1 x timepoints until next iteration x number of outcomes))

# Loop over all patients
for k in range(data_X_test.shape[0]):
    
    print('patient number' + str(k))
    
    # Initializing patient specific lists
    feasible_sol_list_k=[]
    sofa_min_arg_list_k=[]
    best_treat_list_k=[]
    sofa_min_list_k=[]
    feasible_tensors_list_k=[]
    best_treat_int_list_k=[]
    opt_time_list_k=[] 
    best_pred_list_k=[]
    
    violated_sol_list_k=[]
    violated_tensors_list_k=[]
    
    feasible_sol = []
    feasible_res = []
    violated_sol = []
    violated_res = []
    
    
    # variables indicating, whether a patient is treated with specific antibiotic or not
    vancomycin_administrated = -int(data_static_test[k,8]*static_std[-3]+static_mean[-3]) 
    zosyn_administrated = -int(data_static_test[k,9]*static_std[-2]+static_mean[-2]) 
    ceftriaxone_administrated = -int(data_static_test[k,10]*static_std[-1]+static_mean[-1]) 
    
    # Treatment start
    first_ab_tp=(torch.sum(data_treatment_test[k,:,:],axis=-1)>0).nonzero()[0]
    offset=first_ab_tp.item()
    opt_time_list_k.append(offset)
    
    start_pred = offset
    first=True # Indicating whether first iteration or not
    two_antibiotics_allowed=True #Indicating whether two antibiotics are allowed or not due to treatment recommendations (after 48-72h of treatment with 2 antibiotics, one should de-escalated)
    
    if data_static_test[k,static_variables.index('male')]==1:
        se = sideeffects_m
    else:
        se = sideeffects_f 
    
    data_X_test_k = data_X_test[k:k+1,:,:].clone().repeat(3,1,1)
    data_toxic_test_k = data_toxic_test[k:k+1,:,:].clone().repeat(3,1,1)
    data_time_test_k = data_time_test[k:k+1,:,:].clone().repeat(3,1,1)
    data_active_test_k = data_active_test[k:k+1,:,:].clone().repeat(3,1,1)
    data_static_test_k = data_static_test[k:k+1,:].clone().repeat(3,1,1)
    
    # While loop breaks if treatment optimization ends due to decreasing value for 48h OR discharge from ICU 
    while ~np.isnan(start_pred) and ~data_covariables_test[k,start_pred,1].isnan():
        
        creat_dat_k = creat_dat_test[k]
        creat_dat_start_pred = creat_dat_k[creat_dat_k['time_measured']<=start_pred]
        
        # Storing creatinine values for the computation of the Acute Kidney Injury
        if not creat_dat_start_pred.empty:
            min_creat = min(creat_dat_start_pred['creatinine'].values)
            max_creat = max(creat_dat_start_pred['creatinine'].values)
        else:
            min_creat = None
            max_creat = None
        
        # Microbiological data for patient k
        microbiol_dat_k = microbiol_dat_test[k]
        microbiol_dat_start_pred=microbiol_dat_k[(microbiol_dat_k['storetime']<=start_pred) & (microbiol_dat_k['storetime']>-microbiol_time)]
        
        # if first optimization timepoint, prediction horizon is 48 hours (due to guidelines)
        if first:
            pred_horizon=48
            data_covariables_update_k = data_covariables_test[k:k+1,:,:].clone()
            data_treatment_update_k = data_treatment_test[k:k+1,:,:].clone()
            
            data_covariables_update_k = data_covariables_update_k.repeat(2,1,1)
            data_treatment_update_k = data_treatment_update_k.repeat(2,1,1)
            
            # Computing variables indicating duration of treatment with specific antibiotic
            if (torch.flip(data_treatment_update_k[:,:offset,0],dims=(0,1))[0]==0).nonzero(as_tuple=False).shape[0]>0:
                vancomycin_administrated = vancomycin_administrated + (torch.flip(data_treatment_update_k[:,:offset,0],dims=(0,1))[0]==0).nonzero(as_tuple=False)[0].item()                
            else:
                vancomycin_administrated=vancomycin_administrated + offset
            if (torch.flip(data_treatment_update_k[:,:offset,1],dims=(0,1))[0]==0).nonzero(as_tuple=False).shape[0]>0:
                zosyn_administrated = zosyn_administrated + (torch.flip(data_treatment_update_k[:,:offset,1],dims=(0,1))[0]==0).nonzero(as_tuple=False)[0].item()                
            else:
                zosyn_administrated=zosyn_administrated + offset

            if (torch.flip(data_treatment_update_k[:,:offset,2],dims=(0,1))[0]==0).nonzero(as_tuple=False).shape[0]>0:
                ceftriaxone_administrated = ceftriaxone_administrated + (torch.flip(data_treatment_update_k[:,:offset,2],dims=(0,1))[0]==0).nonzero(as_tuple=False)[0].item()                
            else:
                ceftriaxone_administrated=ceftriaxone_administrated + offset
        else:
            pred_horizon=24
            
        # loop over all possible treatment options
        for i in range(len(treatment_options_string)):
            
            # Tensors of all covariables and treatments of patient k
            data_covariables_update_k = data_covariables_test[k:k+1,:,:].clone()
            data_treatment_update_k = data_treatment_test[k:k+1,:,:].clone()
            
            data_covariables_update_k = data_covariables_update_k.repeat(3,1,1)
            data_treatment_update_k = data_treatment_update_k.repeat(3,1,1)
                
            treat_opt_str = treatment_options_string[i]
            treat_opt_int = treatment_options_int[i]
            
            # Setting all antibiotics to zero after start of treatment optimization
            data_covariables_update_k[:,start_pred:,variables_complete.index('Vancomycin')]=0
            data_covariables_update_k[:,start_pred:,variables_complete.index('Piperacillin-Tazobactam')]=0
            data_covariables_update_k[:,start_pred:,variables_complete.index('Ceftriaxon')]=0
            data_treatment_update_k[:,start_pred:,0] = 0
            data_treatment_update_k[:,start_pred:,1] = 0
            data_treatment_update_k[:,start_pred:,2] = 0
            
            # Setting antibiotics to specific treatment configuration i
            for j in range(len(treat_opt_str)):
                data_covariables_update_k[:,start_pred:,variables_complete.index(treat_opt_str[j])]=1
                data_treatment_update_k[:,start_pred:,treat_opt_int[j]] = 1
                
            # Prediction for patient k and treatment i
            pred_X_val, _, _, _, _ = utils_paper.predict_decoder(model, model_decoder, offset=start_pred, max_horizon=pred_horizon, validation_output=data_X_test_k, validation_toxic=data_toxic_test_k, validation_treatments=data_treatment_update_k, covariables=data_covariables_update_k, time_covariates=data_time_test_k, active_entries=data_active_test_k, static=data_static_test_k, rectilinear_index=rectilinear_index, dec_expand=True,sofa_expand=True, med_dec=False, med_dec_start=True)
            
            # Selecting first index of the prediction and de-normalizing variables
            pred_X_val=pred_X_val[0:1,:,:]
            pred_X_val_unsc = pred_X_val[:,:,:].clone()
            pred_X_val_unsc[:,:,0] = pred_X_val[:,:,0]*variables_std[0]+variables_mean[0]
            pred_X_val_unsc[:,:,1] = pred_X_val[:,:,1]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
            pred_X_val_unsc[:,:,2] = pred_X_val[:,:,2]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
            pred_X_val_unsc[:,:,3] = pred_X_val[:,:,3]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
            
            # Missing mask of the side-effects associated variables
            cr_mask=data_covariables_update_k[0,0,variables_complete.index('creatinine_mask')].clone()
            b_mask=data_covariables_update_k[0,0,variables_complete.index('bilirubin_total_mask')].clone()
            alt_mask=data_covariables_update_k[0,0,variables_complete.index('alt_mask')].clone()
            
            # side-effects associated variables
            cr = data_covariables_update_k[0,:start_pred+1,variables_complete.index('creatinine')].clone()*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
            bil = data_covariables_update_k[0,:start_pred+1,variables_complete.index('bilirubin_total')].clone()*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
            alt = data_covariables_update_k[0,:start_pred+1,variables_complete.index('alt')].clone()*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
             
            # If first value was imputed (for the prediction with the TE-CDE), then setting it to nan for the side-effects
            cr[cr_mask==0,0]=np.nan
            bil[b_mask==0,0]=np.nan
            alt[alt_mask==0,0]=np.nan
            
            # fill nan values by propagating last valid observation to next valid
            cr_k=pd.DataFrame(cr.detach().cpu()).ffill(axis=0).iloc[-1]
            bil_k=pd.DataFrame(bil.detach().cpu()).ffill(axis=0).iloc[-1]
            alt_k=pd.DataFrame(alt.detach().cpu()).ffill(axis=0).iloc[-1]

            # Index if side-effects associated thresholds are violated            
            viol = False
            
            # Focussing on dynamic of creatinine value for identifying patients with acute kidney injury (at least) at stage 1
            if not creat_dat_start_pred.empty:
                creat_val = creat_dat_start_pred['creatinine'].values
                if len(creat_val)>0:
                    # Treatment of Vancomycin should be avoided if patient has acute kidney injury
                    if 'Vancomycin' in treat_opt_str:
                        for l in range(len(creat_val)):
                            for j in range(l,len(creat_val)): 
                                if not viol and (creat_val[j]>(creat_val[l]+0.3) or creat_val[j]>(1.5*creat_val[l])):# or creat_val[j]>se[0]):
                                    violated_sol.append(i)
                                    violated_res.append(pred_X_val_unsc)
                                    viol = True
                                    
                        for l in range(pred_X_val_unsc[:,:,1].shape[1]):
                            for j in range(l,pred_X_val_unsc[:,:,1].shape[1]):
                                if not viol and (pred_X_val_unsc[0,j,1]>(pred_X_val_unsc[0,l,1]+0.3) or pred_X_val_unsc[0,j,1]>(1.5*pred_X_val_unsc[0,l,1]) or pred_X_val_unsc[0,j,1]>se[0]):
                                    violated_sol.append(i)
                                    violated_res.append(pred_X_val_unsc)
                                    viol = True
            
            # Treatment should be changed, if pathogens are resistant to it 
            if not viol and not microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='CEFTRIAXONE') & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Ceftriaxon' in treat_opt_str):
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol = True
            
            if not viol and not microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='VANCOMYCIN') & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Vancomycin' in treat_opt_str):
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol = True
            
            if not viol and not microbiol_dat_start_pred[((microbiol_dat_start_pred['ab_name']=='PIPERACILLIN/TAZO') | (microbiol_dat_start_pred['ab_name']=='PIPERACILLIN')) & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Piperacillin-Tazobactam' in treat_opt_str):
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol = True
            # Checking if acute kidney injury is predicted (or observed)
            if not viol and (((cr_k>se[0]).any() or (pred_X_val_unsc[:,:,1]>se[0]).any() or (max_creat is not None and max_creat>se[0]) or (min_creat is not None and (pred_X_val_unsc[:,:,1]>(min_creat+0.3)).any()) or (min_creat is not None and (pred_X_val_unsc[:,:,1]>(1.5*min_creat)).any())) and ('Vancomycin' in treat_opt_str)):# or 'Piperacillin-Tazobactam' in treat_opt_str)):
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol=True
                print('crea')
                
            # Checking if the bilirubin total or alanine transaminse thresholds are violated
            elif not viol and (((bil_k>se[1]).any() or (pred_X_val_unsc[:,:,2]>se[1]).any() or (alt_k>se[2]).any() or (pred_X_val_unsc[:,:,3]>se[2]).any()) and ('Ceftriaxon' in treat_opt_str)):# or 'Piperacillin-Tazobactam' in treat_opt_str)):
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol=True
                print('bil or alt')
            
            # Checking if two antibtiocis are allowed or not
            elif not viol and len(treatment_options_string[i])>1 and not two_antibiotics_allowed:
                violated_sol.append(i)
                violated_res.append(pred_X_val_unsc)
                viol=True
                print('two ab not allowed')
            elif not viol:
                feasible_sol.append(i)
                feasible_res.append(pred_X_val_unsc)
                
        # Appending results to patient specific lists and choice best treatment
        feasible_tensors=torch.stack(feasible_res)
        sofa_min_arg=torch.argmin(feasible_tensors[:,0,-1,0])
        best_treat = treatment_options_string[feasible_sol[sofa_min_arg]]
        sofa_min = torch.min(feasible_tensors[:,0,-1,0])
        best_treat_int=treatment_options_int[feasible_sol[sofa_min_arg]]
        print(feasible_sol)
        print(sofa_min_arg)
        
        print(best_treat)
        print(sofa_min)
        
        feasible_sol_list_k.append(feasible_sol)
        sofa_min_arg_list_k.append(sofa_min_arg)
        best_treat_list_k.append(best_treat)
        sofa_min_list_k.append(sofa_min)
        feasible_tensors_list_k.append(feasible_tensors)
        best_treat_int_list_k.append(best_treat_int)
        best_pred_list_k.append(feasible_tensors[sofa_min_arg,0])
        
        if len(violated_res)>0:
            violated_tensors=torch.stack(violated_res)
        else:
            violated_tensors=[]
        violated_sol_list_k.append(violated_sol)
        violated_tensors_list_k.append(violated_tensors)
        
        feasible_sol = []
        feasible_res = []
        violated_sol = []
        violated_res = []
        
        #checking side conditions and updating covariables
        
        first=False
        
        # Checking for observed side-effects or incoming microbiological data during current optimal treatment iteration
        # Therefore: + pred_horizon
        
        creat_dat_k = creat_dat_test[k]
        creat_dat_start_pred = creat_dat_k[creat_dat_k['time_measured']<=start_pred+pred_horizon]
        if not creat_dat_start_pred.empty:
            min_creat = min(creat_dat_start_pred['creatinine'].values)
            max_creat = max(creat_dat_start_pred['creatinine'].values)
        else:
            min_creat = None
            max_creat = None
        
        microbiol_dat_k = microbiol_dat_test[k]
        microbiol_dat_start_pred=microbiol_dat_k[(microbiol_dat_k['storetime']<=start_pred+pred_horizon) & (microbiol_dat_k['storetime']>-microbiol_time)]
        
        cr_mask=data_covariables_update_k[0,0,variables_complete.index('creatinine_mask')].clone()
        b_mask=data_covariables_update_k[0,0,variables_complete.index('bilirubin_total_mask')].clone()
        alt_mask=data_covariables_update_k[0,0,variables_complete.index('alt_mask')].clone()
        
        cr = data_covariables_update_k[0,:start_pred+pred_horizon+1,variables_complete.index('creatinine')].clone()*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
        bil = data_covariables_update_k[0,:start_pred+pred_horizon+1,variables_complete.index('bilirubin_total')].clone()*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
        alt = data_covariables_update_k[0,:start_pred+pred_horizon+1,variables_complete.index('alt')].clone()*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
            
        cr[cr_mask==0,0]=np.nan
        bil[b_mask==0,0]=np.nan
        alt[alt_mask==0,0]=np.nan
        
        # SOFA-Scores to check, whether treatment can be de-escalated or not
        data_X_test_k_unsc = data_X_test_k.clone()
        data_X_test_k_unsc=data_X_test_k_unsc*variables_std[0]+variables_mean[0]
        
        # Variable indicating, whether treatment has to be change due to a violation of the side-effects associated thresholds is observed (or incoming microbiological data)
        # break_time is set to the timepoint when the treatment has to be changed
        break_time=None
        
        # If conditions are added to find the "earliest" break_time, if there are multiple observed violations of the thresholds
        if not creat_dat_start_pred.empty:
            creat_val = creat_dat_start_pred['creatinine'].values
            if len(creat_val)>0:
                if 'Vancomycin' in best_treat:
                    for l in range(len(creat_val)):
                        for j in range(l,len(creat_val)):
                            if (creat_val[j]>(creat_val[l]+0.3) or creat_val[j]>(1.5*creat_val[l])):# or creat_val[j]>se[0]):
                                if break_time is None or break_time>creat_dat_start_pred.iloc[j]['time_measured']:
                                    break_time=creat_dat_start_pred.iloc[j]['time_measured']
    
        if not microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='CEFTRIAXONE') & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Ceftriaxon' in best_treat):
            if break_time is None or break_time>min(microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='CEFTRIAXONE') & (microbiol_dat_start_pred['interpretation']!='S')]['storetime']):
                break_time=min(microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='CEFTRIAXONE') & (microbiol_dat_start_pred['interpretation']!='S')]['storetime'])
        
        if not microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='VANCOMYCIN') & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Vancomycin' in best_treat):
            if break_time is None or break_time>min(microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='VANCOMYCIN') & (microbiol_dat_start_pred['interpretation']!='S')]['storetime']):
                break_time=min(microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='VANCOMYCIN') & (microbiol_dat_start_pred['interpretation']!='S')]['storetime'])
        
        if not microbiol_dat_start_pred[((microbiol_dat_start_pred['ab_name']=='PIPERACILLIN/TAZO') | (microbiol_dat_start_pred['ab_name']=='PIPERACILLIN')) & (microbiol_dat_start_pred['interpretation']!='S')].empty and ('Piperacillin-Tazobactam' in best_treat):
            if break_time is None or break_time>min(microbiol_dat_start_pred[(microbiol_dat_start_pred['ab_name']=='PIPERACILLIN/TAZO') | (microbiol_dat_start_pred['ab_name']=='PIPERACILLIN') & (microbiol_dat_start_pred['interpretation']!='S')]['storetime']):
                break_time=min(microbiol_dat_start_pred[((microbiol_dat_start_pred['ab_name']=='PIPERACILLIN/TAZO') | (microbiol_dat_start_pred['ab_name']=='PIPERACILLIN')) & (microbiol_dat_start_pred['interpretation']!='S')]['storetime'])
        
        if (cr[start_pred:]>se[0]).any() and ('Vancomycin' in best_treat):# or 'Piperacillin-Tazobactam' in treat_opt_str)):
            if break_time is None or break_time>((cr[start_pred:] > se[0]).nonzero(as_tuple=False)[0]).item()+start_pred:
                break_time = ((cr[start_pred:] > se[0]).nonzero(as_tuple=False)[0]).item()+start_pred
            
        if (bil[start_pred:]>se[1]).any() and ('Ceftriaxon' in best_treat):# or 'Piperacillin-Tazobactam' in treat_opt_str)):
            if break_time is None or break_time>((bil[start_pred:] > se[1]).nonzero(as_tuple=False)[0]).item()+start_pred:
                break_time = ((bil[start_pred:] > se[1]).nonzero(as_tuple=False)[0]).item()+start_pred
        
        if (alt[start_pred:]>se[2]).any() and ('Ceftriaxon' in best_treat):# or 'Piperacillin-Tazobactam' in treat_opt_str)):
            if break_time is None or break_time>((alt[start_pred:] > se[2]).nonzero(as_tuple=False)[0]).item()+start_pred:
                break_time = ((alt[start_pred:] > se[2]).nonzero(as_tuple=False)[0]).item()+start_pred
        
        # Updating variables indicating treatment duration
        if 'Vancomycin' in best_treat:
            if break_time is None:
                vancomycin_administrated=vancomycin_administrated + pred_horizon
            else:
                vancomycin_administrated=vancomycin_administrated + (break_time-start_pred)
        else:
            vancomycin_administrated=0
        if 'Piperacillin-Tazobactam' in best_treat:
            if break_time is None:
                zosyn_administrated=zosyn_administrated + pred_horizon
            else:
                zosyn_administrated=zosyn_administrated + (break_time-start_pred)
        else:
            zosyn_administrated=0
        if 'Ceftriaxon' in best_treat:
            if break_time is None:
                ceftriaxone_administrated=ceftriaxone_administrated + pred_horizon
            else:
                ceftriaxone_administrated=ceftriaxone_administrated + (break_time-start_pred)
        else:
            ceftriaxone_administrated=0
        
        # Checking duration of two administered antibiotics
        if (vancomycin_administrated>72, zosyn_administrated>72, ceftriaxone_administrated>72).count(True)>=2:
            m = min(i for i in [vancomycin_administrated,zosyn_administrated,ceftriaxone_administrated] if i > 72)
            h = m-72
            
            if break_time is None:
                ind = start_pred + pred_horizon-h
            else:
                ind=break_time-h
            
            # Sum of hours antibiotic was administered
            vanc_ad_real=data_treatment_test[k,max(0,ind-h-72):ind-h,0].sum()
            zos_ad_real=data_treatment_test[k,max(0,ind-h-72):ind-h,1].sum()
            cef_ad_real=data_treatment_test[k,max(0,ind-h-72):ind-h,2].sum()
            
            if (vancomycin_administrated>72 and vanc_ad_real>=72, zosyn_administrated>72 and zos_ad_real>=72, ceftriaxone_administrated>72 and cef_ad_real>=72).count(True)>=2:
                two_antibiotics_allowed=False
                
                if vancomycin_administrated>72:
                    vancomycin_administrated=vancomycin_administrated-h
                if zosyn_administrated>72:
                    zosyn_administrated=zosyn_administrated-h
                if ceftriaxone_administrated>72:
                    ceftriaxone_administrated=ceftriaxone_administrated-h
            else:
                h=0
        else:
            h=0
        
        # Indicator variables for the end of a sepsis / de-escalation of treatment
        nosep=False 
        nosep_duration=0 # Duration of decreased SOFA-scores or SOFA-Score lower than 2
        noseptime=None # Timepoint when Sepsis ends
        nosep_sofa=None # Minimum SOFA-Score after decreasing SOFA-Score is observed or lower than 2
        stop_treatment=False # Indicator of whether the treatment can be de-escalated
        
        # Only use SOFA-Scores after SOFA-Score reaches at least 2
        if len(torch.nonzero(data_X_test_k_unsc[0,:,0]>=2))>0:
            
            # Timepoint when SOFA-Score reaches at least 2
            first_index = torch.nonzero(data_X_test_k_unsc[0,:,0]>=2)[0]
            
            # Maximum observed SOFA-Value
            ref_sofa = data_X_test_k_unsc[0,first_index,0]
            
            # Duration of Sepsis
            sep_dur=0
            
            for i in range(0,start_pred+1+pred_horizon):
                
                # break_time-h to check only timepoints BEFORE another treatment change was done
                
                if i>first_index:
                    
                    # Checking, if SOFA-Score decreases by 2 or is smaller than 2 at the current tp for the first time
                    if not nosep and (data_X_test_k_unsc[0,i,0]<2 or data_X_test_k_unsc[0,i,0]<=ref_sofa-2) and (break_time is None or i<=(break_time-h)):
                        noseptime=i
                        nosep=True
                        nosep_sofa = data_X_test_k_unsc[0,i,0] 
                        nosep_duration = 0
                        sep_dur=0
                    
                    # Checking, if "no sepsis" state continues 
                    elif nosep and data_X_test_k_unsc[0,i,0]<=nosep_sofa and (break_time is None or i<=(break_time-h)): 
                        # Increase duration of no Sepsis by 1
                        nosep_duration = nosep_duration + 1
                        
                        # If condition to "ignore" one-off drops of the SOFA-Score while counting no sep duration
                        if data_X_test_k_unsc[0,i,0]<nosep_sofa and sep_dur==0:
                            sep_dur=sep_dur+1
                        elif data_X_test_k_unsc[0,i,0]<nosep_sofa and sep_dur==1:
                            # New minimal SOFA-Score
                            nosep_sofa = data_X_test_k_unsc[0,i,0] 
                            sep_dur=0 
                        else:
                            sep_dur=0
                        
                    # No sepsis before, but NOW (due to increasing SOFA-Score)
                    elif nosep and data_X_test_k_unsc[0,i,0]>nosep_sofa and (break_time is None or i<=(break_time-h)):
                        nosep=False
                        nosep_duration=0
                        noseptime = None
                        nosep_sofa=None
                        ref_sofa = data_X_test_k_unsc[0,i,0]
                    
                    # Sepsis before and increasing SOFA-Score
                    elif not nosep and data_X_test_k_unsc[0,i,0]>ref_sofa and (break_time is None or i<=(break_time-h)):
                        ref_sofa = data_X_test_k_unsc[0,i,0]
                
                # i>=start_pred+1 not necessary
                # If a decreasing SOFA-Score is observed for 48 hours, then treatment should be stopped
                if nosep_duration >=48 and i>=start_pred+1:
                        stop_treatment = True
                        stop_treatment_time = i
                        break;
            
            # Appending the information about de-escalating to the lists
            if stop_treatment: 
                opt_time_list_k.append(stop_treatment_time)
                
                feasible_sol_list_k.append(['nosep'])
                sofa_min_arg_list_k.append(['nosep'])
                best_treat_list_k.append(['nosep'])
                sofa_min_list_k.append(['nosep'])
                feasible_tensors_list_k.append(['nosep'])
                best_treat_int_list_k.append(['nosep'])
                best_pred_list_k.append(['nosep'])
                
                sofa_greater_ind = (data_X_test_k_unsc[0,:,0]>nosep_sofa).nonzero(as_tuple=False)
                
                print('nosepsis at ' + str(stop_treatment_time))
                
                # Necessary to break the while loop
                start_pred=np.nan
                        
            else:
                # Compute start of next optimal treatment iteration
                if break_time is None:
                    start_pred = int(math.ceil(start_pred + pred_horizon-h)) 
                else:
                    start_pred = int(math.ceil(break_time-h))
                
                opt_time_list_k.append(start_pred)
        else:
            if break_time is None:
                start_pred = int(math.ceil(start_pred + pred_horizon-h))
            else:
                start_pred = int(math.ceil(break_time-h))
            
            opt_time_list_k.append(start_pred)
    
    feasible_sol_list_all.append(feasible_sol_list_k)
    sofa_min_arg_list_all.append(sofa_min_arg_list_k)
    best_treat_list_all.append(best_treat_list_k)
    sofa_min_list_all.append(sofa_min_list_k)
    feasible_tensors_list_all.append(feasible_tensors_list_k)
    best_treat_int_list_all.append(best_treat_int_list_k)
    opt_time_list_all.append(opt_time_list_k)
    best_pred_list_all.append(best_pred_list_k)
    
    violated_sol_list_all.append(violated_sol_list_k)
    violated_tensors_list_all.append(violated_tensors_list_k)

with open('/work/wendland/opt_treat/feasible_sol_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(feasible_sol_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/sofa_min_arg_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(sofa_min_arg_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/best_treat_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(best_treat_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/sofa_min_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(sofa_min_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/feasible_tensors_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(feasible_tensors_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
with open('/work/wendland/opt_treat/best_treat_int_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(best_treat_int_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/opt_treat/opt_time_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(opt_time_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/best_pred_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(best_pred_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/violated_sol_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(violated_sol_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('/work/wendland/opt_treat/violated_tensors_list_all_corrected.pkl', 'wb') as handle:
    pickle.dump(violated_tensors_list_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

    

# Counting all patients with violated side-effects associated thresholds
creat_viol_tp_list = []
bil_viol_tp_list = []
alt_viol_tp_list = []

creat_val = data_covariables_test[:,:,variables_complete.index('creatinine')]*variables_std[variables.index("creatinine")]+variables_mean[variables.index("creatinine")]
bil_val = data_covariables_test[:,:,variables_complete.index('bilirubin_total')]*variables_std[variables.index("bilirubin_total")]+variables_mean[variables.index("bilirubin_total")]
alt_val = data_covariables_test[:,:,variables_complete.index('alt')]*variables_std[variables.index("alt")]+variables_mean[variables.index("alt")]


for i in range(creat_val.shape[0]):
    print(i)
    creat_viol_tp = None

    for j in range(creat_val.shape[1]):
        if not torch.isnan(creat_val[i,j]):
            if creat_val[i,j]>2 and (creat_viol_tp is None or j < creat_viol_tp):
                creat_viol_tp = j
            else:
                for k in range(j,creat_val.shape[1]): 
                    if (creat_viol_tp is None or k< creat_viol_tp) and (creat_val[i,k]>(creat_val[i,j]+0.3) or creat_val[i,k]>(1.5*creat_val[i,j])):
                        creat_viol_tp = k
    creat_viol_tp_list.append(creat_viol_tp)
    bil_viol_tp = None

    if data_static_test[i,static_variables.index('male')]==1:
        bil_se = 2.4
    else:
        bil_se = 2.2
    for j in range(bil_val.shape[1]):
        if (bil_viol_tp is None or j < bil_viol_tp) and bil_val[i,j]>bil_se:
            bil_viol_tp = j
    bil_viol_tp_list.append(bil_viol_tp)

    alt_viol_tp = None
    for j in range(alt_val.shape[1]):
        if (alt_viol_tp is None or j < alt_viol_tp) and alt_val[i,j]>280:
            alt_viol_tp = j
    alt_viol_tp_list.append(alt_viol_tp)

# Counting all patients treated with Antibiotics while (after) their side-effects associated thresholds are violated
creat_viol = 0
bil_viol = 0
alt_viol = 0
for i in range(len(creat_viol_tp_list)):
    if creat_viol_tp_list[i] is not None and (data_treatment_test[i,creat_viol_tp_list[i]:,0]>0).any():
        creat_viol = creat_viol + 1
    if bil_viol_tp_list[i] is not None and (data_treatment_test[i,bil_viol_tp_list[i]:,2]>0).any():
        bil_viol = bil_viol + 1
    if alt_viol_tp_list[i] is not None and (data_treatment_test[i,alt_viol_tp_list[i]:,2]>0).any():
        alt_viol = alt_viol + 1
print(creat_viol) #125, 30,6%
print(bil_viol) # 57, 19.1%
print(alt_viol) # 7, 2.3%

# Counting all patients treated with Antibiotics while (after) their side-effects associated thresholds are violated
# AND OptABs optimal treatments do not INCLUDE those Antibiotics BEFORE the violation is observed
creat_viol = 0
bil_viol = 0
alt_viol = 0
for i in range(len(creat_viol_tp_list)):
    if creat_viol_tp_list[i] is not None and (data_treatment_test[i,creat_viol_tp_list[i]:,0]>0).any() and (0,) not in best_treat_int_list_all[i] and (0,1) not in best_treat_int_list_all[i] and (0,2) not in best_treat_int_list_all[i]:
        creat_viol = creat_viol + 1
    if bil_viol_tp_list[i] is not None and (data_treatment_test[i,bil_viol_tp_list[i]:,2]>0).any() and (2,) not in best_treat_int_list_all[i] and (0,2) not in best_treat_int_list_all[i] and (1,2) not in best_treat_int_list_all[i]:
        bil_viol = bil_viol + 1
    if alt_viol_tp_list[i] is not None and (data_treatment_test[i,alt_viol_tp_list[i]:,2]>0).any() and (2,) not in best_treat_int_list_all[i] and (0,2) not in best_treat_int_list_all[i] and (1,2) not in best_treat_int_list_all[i]:
        alt_viol = alt_viol + 1
print(creat_viol) #76, 49/125 =  39,2%
print(bil_viol) # 52
print(alt_viol) # 5
# 7/64 = 10.9%
