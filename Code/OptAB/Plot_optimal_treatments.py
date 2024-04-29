import torch
import numpy as np
import matplotlib.pyplot as plt
import utils_paper
import pandas as pd
import pickle

import itertools


# Plot Helper function
def drawPieMarker(xs, ys, ratios, sizes,ax=None, colors=None, labels=None):
    assert sum(ratios) <= 1, 'sum of ratios needs to be < 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    if colors is not None:
        for color, ratio in zip(colors, ratios):
            this = 2 * np.pi * ratio + previous
            x  = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
            y  = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
            xy = np.column_stack([x, y])
            previous = this
            markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':color})
            if labels is not None:
                markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes), 'facecolor':'white'})
    else:
        for ratio in ratios:
            this = 2 * np.pi * ratio + previous
            x  = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
            y  = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
            xy = np.column_stack([x, y])
            previous = this
            markers.append({'marker':xy, 's':np.abs(xy).max()**2*np.array(sizes)})

    # scatter each of the pie pieces to create pies
    i=0
    for marker in markers:
        if marker['facecolor']!='white':
            ax.scatter(xs, ys, **marker, zorder=10)
        if labels is not None and marker['facecolor']!='white':
            ax.scatter(xs, ys, **marker,label=labels[i], zorder=10)
            i=i+1
        elif labels is not None:
            ax.scatter(xs, ys, **marker, zorder=10)

# Function to create sets based on lists
def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s,r) for r in range(len(s) +1))

with open('/work/wendland/data_prep_three/sepsis_all_1_lab.pkl', 'rb') as handle:
    data = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_keys.pkl', 'rb') as handle:
    key_times = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_static.pkl', 'rb') as handle:
    static_tensor = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_indices_train.pkl', 'rb') as handle:
    indices_train = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_indices_test.pkl', 'rb') as handle:
    indices_test = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_variables.pkl', 'rb') as handle:
    variables = pickle.load(handle)

with open('/work/wendland/data_prep_three/creat_dat.pkl', 'rb') as handle:
    creat_dat = pickle.load(handle)

with open('/work/wendland/data_prep_three/microbiol_dat_.pkl', 'rb') as handle:
    microbiol_dat = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_static_mean.pkl', 'rb') as handle:
    static_mean = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_static_std.pkl', 'rb') as handle:
    static_std = pickle.load(handle)

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

with open('/work/wendland/opt_trt_update/sofa_min_arg_list_all_corrected.pkl', 'rb') as handle:
    sofa_min_arg_list_all = pickle.load(handle)

with open('/work/wendland/data_prep_three/pred_X_val_batch13.pkl', 'rb') as handle:
    onesteppred = pickle.load(handle)

#### "Pre-Processing of data"

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

#Extracting the covariables
data_covariables_test = data[:,:list(key_times.keys())[-1],:].clone()[indices_test].to(device)

#Normalizing the missing masks to one
time_max = data.shape[1]
data_covariables_test[:,:,len(variables)+1:] = data_covariables_test[:,:,len(variables)+1:]/time_max
data_covariables_test[:,:,0] = data_covariables_test[:,:,0]/time_max

# Selection of training and test data
data_time_test = data[:,:list(key_times.keys())[-1],0:1][indices_test].to(device)
data_active_test = ~data[:,list(key_times.values()),1:2].isnan()[indices_test].to(device)
data_static_test=static_tensor[indices_test].to(device,dtype=torch.float32)    


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


data_covariables_test[data_covariables_test[:,0,variables_complete.index('creatinine_mask')]==0,0,variables_complete.index('creatinine')]=np.nan
data_covariables_test[data_covariables_test[:,0,variables_complete.index('bilirubin_total_mask')]==0,0,variables_complete.index('bilirubin_total')]=np.nan
data_covariables_test[data_covariables_test[:,0,variables_complete.index('alt_mask')]==0,0,variables_complete.index('alt')]=np.nan

creat_val = data_covariables_test[:,:,variables_complete.index('creatinine')]*variables_std[variables.index("creatinine")]+variables_mean[variables.index("creatinine")]
bil_val = data_covariables_test[:,:,variables_complete.index('bilirubin_total')]*variables_std[variables.index("bilirubin_total")]+variables_mean[variables.index("bilirubin_total")]
alt_val = data_covariables_test[:,:,variables_complete.index('alt')]*variables_std[variables.index("alt")]+variables_mean[variables.index("alt")]

# Number of patients with at least one observed value of the side-effects associated laboratory values

torch.sum(torch.sum(~torch.isnan(creat_val),axis=1)>0) #697 values
torch.sum(torch.sum(~torch.isnan(bil_val),axis=1)>0) #404 values
torch.sum(torch.sum(~torch.isnan(alt_val),axis=1)>0) #399 values

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


data_thresh = ((0-variables_mean)/variables_std)[[variables.index("SOFA"),variables.index("creatinine"),variables.index("bilirubin_total"),variables.index("alt")]]
data_treat_thresh = None

# Initializing and loading the Encoder
model = utils_paper.NeuralCDE(input_channels=data.shape[2], hidden_channels=hidden_channels, hidden_states=hidden_states, output_channels=4, treatment_options=3, activation = activation, num_depth=num_depth, interpolation="linear", pos=True, thresh=data_thresh, pred_comp=pred_comp, pred_act=pred_act, pred_states=pred_states, pred_depth=pred_depth,static_dim=len(static_variables))
model=model.to(model.device)
model.load_state_dict(torch.load('/work/wendland/Trained_Encoder.pth'))


rectilinear_index=0

# Computing one step prediction
onesteppred, _, _, _, _ = utils_paper.predict(model,train_output=data_X_test, train_toxic=data_toxic_test, train_treatments=data_treatment_test, covariables=data_covariables_test, active_entries=data_active_test, static=data_static_test, rectilinear_index=rectilinear_index)    


treatment_options_string = list(powerset(['Vancomycin','Piperacillin-Tazobactam','Ceftriaxon']))[1:-1]
treatment_options_int = list(powerset([0,1,2]))[1:-1]

trt_string = ['Vancomycin','Pip/Taz','Ceftriaxone','Vanco, Pip/Taz','Vanco, Cef','Pip/Taz, Cef']

plot_1_step = True # Variable Indicating whether 1-hour predictions are plotted or not
round_ = False # Variable indicating, whether SOFA-Scores are rounded


colorlist = [(0,1,0,0.6),(0,0,1,0.7),(0,1,1,0.75),(108/255,122/255,137/255,0.8),(1,0,0,0.85),(243/255,156/255,18/255,1)]

for i in range(len(best_pred_list_all)):
    
    print(i)
    
    opt_time = opt_time_list_all[i]
    
    # Loop selecting all predictions of optimal treatment prediction
    all_trts = []
    all_trts_fs=[]
    for j in range(len(violated_sol_list_all[i])):
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
        
    # manually adjusting some plot parameters due to break in treatment optimization
    if len(opt_time)==2 or i == 483:
        step=2
    elif i==788 or i==15:
        step=3
    elif i==761 or i==846:
        step=5
    elif i==755 or i==223:
        step=8
    elif len(opt_time)==3:
        step=4
    elif len(opt_time)==4:
        step=5
    elif len(opt_time)==5:
        step=6
    elif len(opt_time)==6:
        step=8
    else:
        step=10
        
    # Plotting SOFA-Score based on observed thresholds
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,0].detach())
            else:
                y = all_trts[j][k][0][:,0].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,1]*variables_std[0]+variables_mean[0]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("SOFA-Score",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("SOFA-Score",fontsize=16)
    
    plt.plot(x[::step],y[::step],marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    
    for j in range(0,y.shape[0],step):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
    
    nan_index=data_covariables_test[i,:,1].isnan().nonzero(as_tuple=True)[0][0]
    x=range(1,onesteppred.shape[1]+1)
    y=onesteppred[i,:,0]*variables_std[0]+variables_mean[0]
    y=y.detach().numpy()
    y[nan_index:]=np.nan
    plt.plot(x[::step],y[::step],color='black')
            
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_SOFA' + str(i) +'.png')    
    
    # Plotting SOFA-Score with manually adjusted y-axis
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,0].detach())
            else:
                y = all_trts[j][k][0][:,0].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,1]*variables_std[0]+variables_mean[0]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("SOFA-Score",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("SOFA-Score",fontsize=16)
    
    plt.plot(x[::step],y[::step],marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    
    for j in range(0,y.shape[0],step):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
    
    nan_index=data_covariables_test[i,:,1].isnan().nonzero(as_tuple=True)[0][0]
    x=range(1,onesteppred.shape[1]+1)
    y=onesteppred[i,:,0]*variables_std[0]+variables_mean[0]
    y=y.detach().numpy()
    y[nan_index:]=np.nan
    plt.plot(x[::step],y[::step],color='black')
            
    if i==666:
        plt.ylim(2,8)
    
    elif i==604 or i==21 or i==584 or i==182:
        plt.ylim(-0.25,6)
    elif i==458:
        plt.ylim(-0.25,8)
        
    else:
        plt.ylim(-0.25,14)
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_SOFA_lim' + str(i) +'.png')
    
    # Plotting creatinine based on observed thresholds
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,1].detach())
            else:
                y = all_trts[j][k][0][:,1].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('creatinine')]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Creatinine",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("mg/dl",fontsize=16)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    
    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_creatinine_' + str(i) +'.png')    
    
    # Plotting creatinine with manually adjusted y axis
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,1].detach())
            else:
                y = all_trts[j][k][0][:,1].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('creatinine')]*variables_std[variables.index('creatinine')]+variables_mean[variables.index('creatinine')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Creatinine",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("mg/dl",fontsize=16)
    
    if i==679 or i==124 or i==615 or i==666 or i==311:
        plt.ylim(-0.25,17)
    elif i==667 or i==215:
        plt.ylim(-0.25,8)
    elif i==21 or i==70 or i==604 or i==584:
        plt.ylim(-0.25,3)
    elif i==182:
        plt.ylim(-0.25,3)
    else:
        plt.ylim(-0.25,6)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']

    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_creatinine_lim_lim_' + str(i) +'.png')    
    
    # Plotting bilirubin total based on observed thresholds
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,2].detach())
            else:
                y = all_trts[j][k][0][:,2].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('bilirubin_total')]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Bilirubin",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("mg/dl",fontsize=16)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']

    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
            

    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_bilirubin_' + str(i) +'.png')
    
    # Plotting bilirubin total with manually adjusted y axis
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,2].detach())
            else:
                y = all_trts[j][k][0][:,2].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('bilirubin_total')]*variables_std[variables.index('bilirubin_total')]+variables_mean[variables.index('bilirubin_total')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("Bilirubin",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("mg/dl",fontsize=16)
    if i==604:
        plt.ylim(-0.25,4)
    
    elif i==21 or i==584 or i==182 or i==458:
        plt.ylim(-0.25,4)
    
    else:
        plt.ylim(-0.25,12)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    
    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_bilirubin_lim_lim_' + str(i) +'.png')
    
    # Plotting alanine transaminase based on observed thresholds
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,3].detach())
            else:
                y = all_trts[j][k][0][:,3].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])

    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('alt')]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    #if ['nosep'] in feasible_sol_list_all[i]:
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("ALT",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    plt.ylabel("IU/L",fontsize=16)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    if i==89 or i==311:
        xs=[20,20,20]
        ys=[400,400,400]
    else:
        xs=[x[0],x[0],x[0]]
        ys=[y[0],y[0],y[0]]
    drawPieMarker(xs=xs,
                  ys=ys,
                  ratios=[.33, .33, .33],
                  sizes=[110, 110, 110],
                  colors=['k', 'k', 'k'],
                  labels=labels,ax=ax)
    drawPieMarker(xs=xs,
                  ys=ys,
                  ratios=[.33, .33, .33],
                  sizes=[110, 110, 110],
                  colors=['white', 'white', 'white'],
                  ax=ax)
    
    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
            
    plt.plot(0,0,color='black',label='1-step pred')
    plt.plot(0,0,color='white')  
            
    plt.legend(fontsize=12)
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_alt_' + str(i) +'.png')
    
    # Plotting alanina transaminase with manually adjusted y axis
    
    fig,ax=plt.subplots(figsize=(5.5,5.5),dpi=600)

    for j in range(len(all_trts)):
        x=range(opt_time[j]+2,opt_time[j]+2 + all_trts[j][0].shape[1])
        for k in range(len(all_trts[j])):
            
            if round_:
                y=torch.round(all_trts[j][k][0][:,3].detach())
            else:
                y = all_trts[j][k][0][:,3].detach()
            
            if j==0 and k==0:
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='k',label='Violated')
                plt.plot(x[0],y[0],linestyle='None',marker='x',color='white')
                
            if j==0:
                plt.plot(x[0],y[0],color=colorlist[k],label=trt_string[k])
                plt.plot(x[0],y[0],color='white')
                
            if all_trts_fs[j][k]=='feasible' or all_trts_fs[j][k]=='best':
                plt.plot(x[::step],y[::step],color=colorlist[k])
            else:
                plt.plot(x[::step],y[::step],linestyle='None',marker='x',color=colorlist[k])
    for j in opt_time[:-1]:
        if j == opt_time[0]:
            plt.axvline(j,color='black')#,label='Treat. Eval.')
        else:
            plt.axvline(j,color='black')

    x=range(data_X_test.shape[1])
    y=data_covariables_test[i,:,variables_complete.index('alt')]*variables_std[variables.index('alt')]+variables_mean[variables.index('alt')]
    if i==15 or i==788:
        y[opt_time[-1]:]=np.nan
    if ['nosep'] in feasible_sol_list_all and i != 251:
        plt.xlim(x[0]-5,(torch.isnan(data_X_test[i,:,0])).nonzero(as_tuple=True)[0][0])
    elif i==251 or i==846:
        plt.xticks(range(x[0],opt_time[-1]+48,20),fontsize=13)
    else:
        plt.xlim(x[0]-5,opt_time[-1]+5)
    if len(opt_time)<3:
        plt.xticks(range(x[0],opt_time[0]+50,10),fontsize=13)
    else:
        plt.xticks(range(x[0],opt_time[-1],20),fontsize=13)
    plt.yticks(fontsize=13)
    plt.title("ALT",fontsize=16)
    plt.xlabel("Time in hours",fontsize=16)
    if i== 182:
        plt.ylabel("IU/L",fontsize=16,labelpad=-2)
    else:
        plt.ylabel("IU/L",fontsize=16,labelpad=-8)
        
    
    if i==604:
        plt.ylim(-0.25,1500)
    elif i==21 or i==584 or i==182:
        plt.ylim(-0.25,200)
    elif i==458:
        plt.ylim(-0.25,500)
    else:
        plt.ylim(-0.25,1500)
    
    plt.plot(x,y,marker='o',linestyle='None',fillstyle='none',markersize=np.sqrt(110),color='k',label='Observation')
    
    labels = ['Vancomycin','Pip/Taz','Ceftriaxone']
    if i==89 or i==311:
        xs=[20,20,20]
        ys=[400,400,400]
    else:
        xs=[x[0],x[0],x[0]]
        ys=[y[0],y[0],y[0]]
    drawPieMarker(xs=xs,
                  ys=ys,
                  ratios=[.33, .33, .33],
                  sizes=[110, 110, 110],
                  colors=['k', 'k', 'k'],
                  labels=labels,ax=ax)
    drawPieMarker(xs=xs,
                  ys=ys,
                  ratios=[.33, .33, .33],
                  sizes=[110, 110, 110],
                  colors=['white', 'white', 'white'],
                  ax=ax)
    
    for j in range(0,y.shape[0]):
        best_treat_int=[x[j]>=l for l in opt_time]
        if data_treatment_test[i,x[j],0]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['k', 'white', 'white'])
        if data_treatment_test[i,x[j],1]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                          colors=['white', 'k','white'])
        if data_treatment_test[i,x[j],2]==1:
            drawPieMarker(xs=x[j],
                          ys=y[j].numpy(),
                          ratios=[.33, .33, .33],
                          sizes=[110],ax=ax,
                           colors=['white', 'white', 'k'])
            
    plt.plot(0,0,color='black',label='1-step pred')
    plt.plot(0,0,color='white')    
        
    plt.legend(loc='upper right',fontsize=12)
    
    plt.savefig('/work/wendland/GitHub/TE-CDE-main/Bilder_treat_opt/test_onepatient_alt_lim_lim_' + str(i) +'.png')
    