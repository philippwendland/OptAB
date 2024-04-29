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
    
    
    # Hyperparameter and their distributions
    hidden_channels = trial.suggest_int('hidden_channels',1,30)
    batch_size=trial.suggest_categorical('batch_size',[500,1000,2000])
    hidden_states = trial.suggest_int('hidden_states',16,1000)
    lr = trial.suggest_uniform('lr',0.0001,0.01)
    activation = trial.suggest_categorical('activation',['leakyrelu','tanh','sigmoid','identity'])
    num_depth = trial.suggest_int('numdepth',1,20)
    
    pred_comp=True
    pred_act = trial.suggest_categorical('pred_act',['leakyrelu','tanh','sigmoid','identity'])
    pred_states = trial.suggest_int('pred_states',16,1000)
    pred_depth = trial.suggest_int('preddepth',1,6)
    
    # Threshold for the model to compute only positive outputs (via softplus)
    data_thresh = ((0-variables_mean)/variables_std)[[variables.index("SOFA"),variables.index("creatinine"),variables.index("bilirubin_total"),variables.index("alt")]]
    
    # Initializing the Encoder
    model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables))
    model=model.to(model.device)
    
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
    data_active = ~data[:,list(key_times.values()),1:2].isnan()[indices_train].to(model.device)
    data_static=static_tensor[indices_train].to(model.device,dtype=torch.float32)
    
    train_index = np.random.choice(a=list(range(data_X.shape[0])),size=int(data_X.shape[0]*0.8),replace=False)
    test_index = [i for i in list(range(data_X.shape[0])) if i not in indices_train]
    
    # Train/Test Split for training
    data_X_train =data_X[train_index]
    data_toxic_train=data_toxic[train_index]
    data_treatment_train=data_treatment[train_index]
    data_covariables_train = data_covariables[train_index]
    data_active_train = data_active[train_index]
    data_static_train = data_static[train_index]
    
    data_X_test =data_X[test_index]
    data_toxic_test=data_toxic[test_index]
    data_treatment_test=data_treatment[test_index]
    data_covariables_test = data_covariables[test_index]
    data_active_test = data_active[test_index]
    data_static_test = data_static[test_index]
    
    ### End preprocessing
    
    # Training for specific hyperparameterconfigurations
    try:
        loss = utils_paper.train(model, weight_loss=True, lr=lr, batch_size=batch_size, patience=10, delta=0.0001, max_epochs=1000, train_output=data_X_train, train_toxic=data_toxic_train, train_treatments=data_treatment_train, covariables=data_covariables_train, active_entries=data_active_train, validation_output=data_X_test, validation_toxic=data_toxic_test, validation_treatments=data_treatment_test, covariables_val=data_covariables_test, active_entries_val=data_active_test, static=data_static_train,static_val=data_static_test,rectilinear_index=rectilinear_index,early_stop_path=str('compdepth_static_batch' + str(trial.number) + '.pth'))
    
    except Exception as e:
        print(e)
        loss = np.nan

    print(trial.number)
    print(loss)
    print(trial.params)
    
    torch.save(model.state_dict(), str("/work/wendland/tecde_three_compdepth/final_model_static_hypopt_batch" + str(trial.number) + ".pth"))
    
    return loss
    
    
study = optuna.create_study()

# n_trials is the stopping criterion, n_jobs is the number of parallel used cpus/gpus
study.optimize(time_objective, n_trials=20, n_jobs=1)


print("Number of finished trials: ", len(study.trials))
    
print("Best trial:")
trial = study.best_trial

train_dir="/work/wendland/tecde_three_compdepth"

print("  rec_loss: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

    file = open(train_dir+'/tries_lat_static.txt', 'a')
    file.write("    {}: {}".format(key, value))
    file.write("\n")
file.write(str(trial.user_attrs))
file.write("rec_loss: "+str(trial.value))
file.write("\n"+"------------"+"\n")
file.close()

dic = dict(trial.params)
dic['value'] = trial.value

df = pd.DataFrame.from_dict(data=dic,orient='index').to_csv(train_dir + '/tries_lat_static.csv',header=False)

# Saving and exporting study
joblib.dump(study, "/work/wendland/tecde_three_compdepth/study_static_batch.pkl")


