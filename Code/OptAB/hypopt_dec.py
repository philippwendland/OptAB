import torch
import numpy as np
import utils_paper

import optuna
import joblib
import pandas as pd
import pickle

def time_objective(trial):
    
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
        
    with open('/work/wendland/tecde_three_compdepth/sepsis_all_1_indices_train.pkl', 'rb') as handle:
        indices_train = pickle.load(handle)
        
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
    model.load_state_dict(torch.load('/work/wendland/Trained_Encoder.pth'))

    # Hyperparameters and their distributions
    hidden_channels_dec = trial.suggest_int('hidden_channels_dec',1,30)
    batch_size_dec=trial.suggest_categorical('batch_size_dec',[1000,2000])
    hidden_states_dec = trial.suggest_int('hidden_states_dec',16,1000)
    lr_dec = trial.suggest_uniform('lr_dec',0.0001,0.01)
    activation_dec = trial.suggest_categorical('activation_dec',['leakyrelu','tanh','sigmoid','identity'])
    num_depth_dec = trial.suggest_int('numdepth_dec',1,20)
    
    pred_comp=True
    pred_act_dec = trial.suggest_categorical('pred_act_dec',['leakyrelu','tanh','sigmoid','identity'])
    pred_states_dec = trial.suggest_int('pred_states_dec',16,1000)
    pred_depth_dec = trial.suggest_int('preddepth',1,6)
    
    offset=0
    
    # determined by data
    output_channels=4 # Number of (all) outcomes 
    input_channels_dec=1 # Only time as control
    z0_hidden_dimension_dec = hidden_channels + 1 + static_tensor.shape[-1] + 6
    
    # Initializing Decoder
    model_decoder = utils_paper.NeuralCDE(input_channels=input_channels_dec,hidden_channels=hidden_channels_dec, hidden_states=hidden_states_dec,output_channels=output_channels, z0_dimension_dec=z0_hidden_dimension_dec,activation=activation_dec,num_depth=num_depth_dec, pos=True, thresh=data_thresh, pred_comp=True, pred_act=pred_act_dec, pred_states=pred_states_dec, pred_depth=pred_depth_dec, treatment_options=3)
    model_decoder=model_decoder.to(model_decoder.device)
    
    rectilinear_index=0
    
    #### "Pre-Processing of data"

    # "Cutting" key_times after last observed timepoint of TRAINING (!) data (in this split similar)
    data_active_overall = (~data[:,list(key_times.values()),1:2].isnan()[indices_train])
    key_times_index = np.array(list(key_times.keys()))[:len(data_active_overall[:,:,0].any(0)) - list(data_active_overall[:,:,0].any(0))[::-1].index(True)]
    key_times={list(key_times.keys())[x]: key_times[x] for x in key_times_index}
    
    #Extracting outcome and side-effects from data and setting device
    data_X = data[:,list(key_times.values())[1:],1:2][indices_train].to(model.device)
    data_toxic=data[:,list(key_times.values())[1:],[i=="creatinine" for i in variables_complete]]
    data_toxic=data_toxic[:,:,None][indices_train].to(model.device)
    data_toxic2=data[:,list(key_times.values())[1:],[i=="bilirubin_total" for i in variables_complete]]
    data_toxic2=data_toxic2[:,:,None][indices_train].to(model.device)
    data_toxic3=data[:,list(key_times.values())[1:],[i=="alt" for i in variables_complete]]
    data_toxic3=data_toxic3[:,:,None][indices_train].to(model.device)
    data_toxic = torch.cat([data_toxic,data_toxic2,data_toxic3],axis=-1)
    
    #Extracting treatments and side-effects from data and setting device
    data_treatment=data[:,list(key_times.values()),[i=="Vancomycin" for i in variables_complete]]
    data_treatment=data_treatment[:,:,None][indices_train].to(model.device)
    data_treatment2=data[:,list(key_times.values()),[i=="Piperacillin-Tazobactam" for i in variables_complete]]
    data_treatment2=data_treatment2[:,:,None][indices_train].to(model.device)
    data_treatment3=data[:,list(key_times.values()),[i=="Ceftriaxon" for i in variables_complete]]
    data_treatment3=data_treatment3[:,:,None][indices_train].to(model.device)
    data_treatment = torch.cat([data_treatment,data_treatment2,data_treatment3],axis=-1)
    
    #Extracting the covariables
    data_covariables = data[:,:list(key_times.keys())[-1],:].clone()[indices_train].to(model.device)
    
    #Normalizing the missing masks to  one
    time_max = data.shape[1]
    data_covariables[:,:,len(variables)+1:] = data_covariables[:,:,len(variables)+1:]/time_max
    data_covariables[:,:,0] = data_covariables[:,:,0]/time_max
    
    # Selection of training and test data
    data_time = data[:,:list(key_times.keys())[-1],0:1][indices_train].to(model.device)
    data_active = ~data[:,list(key_times.values()),1:2].isnan()[indices_train].to(model.device)
    data_static=static_tensor[indices_train].to(model.device,dtype=torch.float32)
    
    train_index = np.random.choice(a=list(range(data_X.shape[0])),size=int(data_X.shape[0]*0.8),replace=False)
    test_index = [i for i in list(range(data_X.shape[0])) if i not in indices_train]
    
    # Train/Test Split for training
    data_X_train =data_X[train_index]
    data_toxic_train=data_toxic[train_index]
    data_treatment_train=data_treatment[train_index]
    data_covariables_train = data_covariables[train_index]
    data_time_train = data_time[train_index]  
    data_active_train = data_active[train_index]
    data_static_train = data_static[train_index]
    
    data_X_test =data_X[test_index]
    data_toxic_test=data_toxic[test_index]
    data_treatment_test=data_treatment[test_index]
    data_covariables_test = data_covariables[test_index]
    data_time_test = data_time[test_index]
    data_active_test = data_active[test_index]
    data_static_test = data_static[test_index]
    
    ### End preprocessing
    
    # "Stratified" Sampling the offsets used for the training of the Decoder
    
    torch.sum(~data_X_train[:,:].isnan(),dim=0)/data_X_train.shape[0]
    a=list((torch.sum(~data_X_train[:,:].isnan(),dim=0)/(data_X_train.shape[0]))>0.1)
    max_horizon=a.index(False)
    
    b=list(range(max_horizon))
    offset_train_list=[]
    for i in range(0,len(b),10):
        if i<=20:
            c=np.random.choice(b[i:i+10],size=2,replace=False)
            offset_train_list.append(c[0])
            offset_train_list.append(c[1])
        else:
            offset_train_list.append(np.random.choice(b[i:i+10],replace=False))
    offset_train_list.sort()
    
    
    # Training for specific hyperparameterconfigurations
    try:
        loss = utils_paper.train_dec_offset(model,model_decoder, offset=offset, max_horizon=max_horizon, lr=lr_dec, batch_size=batch_size_dec, patience=10, delta=0.0001, max_epochs=1000, weight_loss=True, train_output=data_X_train, train_toxic=data_toxic_train, train_treatments=data_treatment_train, covariables=data_covariables_train, time_covariates=data_time_train, active_entries=data_active_train, static=data_static_train, rectilinear_index=rectilinear_index, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables_val=data_covariables_test, validation_time_covariates=data_time_test, active_entries_val=data_active_test,static_val=data_static_test, offset_train_list=offset_train_list, offset_val_list=None,early_stop_path=str('compdepth_dec_nomed_batch_encbatch13_static' + str(trial.number) + '.pth'), dec_expand=True,sofa_expand=True,med_dec=False,med_dec_start=True)

    except Exception as e:
        print(e)
        loss = np.nan
    
    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model_decoder.state_dict(), str("/work/wendland/tecde_three_compdepth/final_model_nomed_batch_decoder_hypopt" + str(trial.number) + ".pth"))

    return loss

# Loop to save the hyperparameters
for i in range(20):
    if i>0:
        study = joblib.load("/work/wendland/tecde_three_compdepth/study_dec_nomed_batch_encbatch13_static.pkl")
    else:
        study=optuna.create_study()
        
    # Initializing hyperparameteroptimization by "good" hyperparameters of the Encoder
        
    if i==0:
        study.enqueue_trial({'hidden_channels_dec': 30, 'batch_size_dec': 1000, 'hidden_states_dec': 544, 'lr_dec': 0.006304913877025918, 'activation_dec': 'sigmoid', 'numdepth_dec': 10, 'pred_act_dec': 'leakyrelu', 'pred_states_dec': 923, "preddepth": 1})
    elif i==1:
        study.enqueue_trial({'hidden_channels_dec': 8, 'batch_size_dec': 1000, 'hidden_states_dec': 391, 'lr_dec': 0.00966781540535167, 'activation_dec': 'sigmoid', 'numdepth_dec': 20, 'pred_act_dec': 'tanh', 'pred_states_dec': 21, "preddepth": 1})
    elif i==2:
        study.enqueue_trial({'hidden_channels_dec': 4, 'batch_size_dec': 1000, 'hidden_states_dec': 521, 'lr_dec': 0.004077174360596377, 'activation_dec': 'sigmoid', 'numdepth_dec': 9, 'pred_act_dec': 'identity', 'pred_states_dec': 680, "preddepth": 1})
    elif i==3:
        study.enqueue_trial({"hidden_channels_dec": 17, 'batch_size_dec': 1000, "hidden_states_dec": 33, "lr_dec": 0.0050688746606452565, "activation_dec": 'tanh', "numdepth_dec": 15, "pred_act_dec": 'tanh', "pred_states_dec": 128, "preddepth": 1})
    elif i==4:
        study.enqueue_trial({"hidden_channels_dec": 20, 'batch_size_dec': 1000, "hidden_states_dec": 804, "lr_dec": 0.000308314286773333, "activation_dec": 'sigmoid', "numdepth_dec": 1, "pred_act_dec": 'tanh', "pred_states_dec": 128, "preddepth": 1})
    elif i==5:
        study.enqueue_trial({"hidden_channels_dec": 22, 'batch_size_dec': 1000, "hidden_states_dec": 802, "lr_dec": 0.0016227982436909543, "activation_dec": 'leakyrelu', "numdepth_dec": 13, "pred_act_dec": 'leakyrelu', "pred_states_dec": 798, "preddepth": 1})
    elif i==6:
        study.enqueue_trial({"hidden_channels_dec": 24, 'batch_size_dec': 1000, "hidden_states_dec": 316, "lr_dec": 0.0010974778722903065, "activation_dec": 'leakyrelu', "numdepth_dec": 16, "pred_act_dec": 'leakyrelu', "pred_states_dec": 91, "preddepth": 3})
    elif i==7:
        study.enqueue_trial({"hidden_channels_dec": 9, 'batch_size_dec': 1000, "hidden_states_dec": 586, "lr_dec": 0.0016491513237022039, "activation_dec": 'sigmoid', "numdepth_dec": 10, "pred_act_dec": 'leakyrelu', "pred_states": 723, "preddepth": 1})
    elif i==8:
        study.enqueue_trial({"hidden_channels_dec": 16, 'batch_size_dec': 1000, "hidden_states_dec": 578, "lr_dec": 0.004239690693777566, "activation_dec": 'leakyrelu', "numdepth_dec": 2, "pred_act_dec": 'tanh', "pred_states_dec": 128, "preddepth": 1})
    elif i==9:
        study.enqueue_trial({"hidden_channels_dec": 12, 'batch_size_dec': 1000, "hidden_states_dec": 389, "lr_dec": 0.004774454736584454, "activation_dec": 'leakyrelu', "numdepth_dec": 2, "pred_act_dec": 'tanh', "pred_states_dec": 128, "preddepth": 1})
    elif i==10:
        study.enqueue_trial({"hidden_channels_dec": 28, 'batch_size_dec': 1000, "hidden_states_dec": 109, "lr_dec": 0.0021391309181124276, "activation_dec": 'tanh', "numdepth_dec": 6, "pred_act_dec": 'tanh', "pred_states_dec": 128, "preddepth": 1})
    elif i==11:
        study.enqueue_trial({"hidden_channels_dec": 15, 'batch_size_dec': 1000, "hidden_states_dec": 749, "lr_dec": 0.008449001639402305, "activation_dec": 'leakyrelu', "numdepth_dec": 3, "pred_act_dec": 'leakyrelu', "pred_states_dec": 245, "preddepth": 1})
    elif i==12:
        study.enqueue_trial({"hidden_channels_dec": 24, 'batch_size_dec': 1000, "hidden_states_dec": 316, "lr_dec": 0.0010974778722903065, "activation_dec": 'leakyrelu', "numdepth_dec": 16, "pred_act_dec": 'leakyrelu', "pred_states_dec": 91, "preddepth": 3})
    elif i==13:
        study.enqueue_trial({"hidden_channels_dec": 21, 'batch_size_dec': 1000, "hidden_states_dec": 490, "lr_dec": 0.0019496180041255972, "activation_dec": 'sigmoid', "numdepth_dec": 14, "pred_act_dec": 'tanh', "pred_states_dec": 832, "preddepth": 5})
    elif i==14:
        study.enqueue_trial({"hidden_channels_dec": 29, 'batch_size_dec': 1000, "hidden_states_dec": 275, "lr_dec": 0.00481144397375774, "activation_dec": 'sigmoid', "numdepth_dec": 10, "pred_act_dec": 'leakyrelu', "pred_states_dec": 490, "preddepth": 6})
    elif i==15:
        study.enqueue_trial({"hidden_channels_dec": 10, 'batch_size_dec': 1000, "hidden_states_dec": 182, "lr_dec": 0.001017859129348864, "activation_dec": 'sigmoid', "numdepth_dec": 12, "pred_act_dec": 'tanh', "pred_states_dec": 639, "preddepth": 1})

    
    study.optimize(time_objective, n_trials=1, n_jobs=1)
    
    train_dir="/work/wendland/tecde_three_compdepth"
    
    joblib.dump(study, "/work/wendland/tecde_three_compdepth/study_dec_nomed_batch_encbatch13_static.pkl")



