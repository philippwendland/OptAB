import pandas as pd
import pickle
import numpy as np

drugs = pd.read_csv('/work/wendland/AmsterdamUMCdb-v1.0.2/drugitems.csv', encoding='latin-1')

sepsis3 = pd.read_csv('/work/wendland/Amsterdamdata/sepsis3_def.csv')

with open('/work/wendland/data_prep_three/sepsis_all_1_static_mean.pkl', 'rb') as handle:
    static_mean = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_static_std.pkl', 'rb') as handle:
    static_std = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_static_variables.pkl', 'rb') as handle:
    static_variables = pickle.load(handle)

sepsis_ids = list(sepsis3["admissionid"])


#### Demographics
admissions = pd.read_csv('/work/wendland/AmsterdamUMCdb-v1.0.2/admissions.csv', encoding='latin-1')

admissions_sep = admissions[admissions['admissionid'].isin(sepsis_ids)]

#### DRUGS
drugs = drugs[drugs["admissionid"].isin(sepsis_ids)]
a=drugs[drugs["ordercategoryid"].isin([15])]
a2=drugs[drugs["ordercategoryid"].isin([15,21])]
a2 = a2[['SDD' not in i for i in a2["item"]]]

# ordercategoryid: 15 = 'Injecties Antimicrobiele middelen', Antibiotika, 65 'Second injection'
# 21 = 'Niet iv Antimicrobiele middelen'
# 'Vancomycine', Ceftriaxon (Rocephin), 'Piperacilline (Pipcil)'

second_injection =  drugs[(drugs["ordercategoryid"]==65) & drugs["admissionid"].isin(sepsis_ids)]
second_injection = second_injection[second_injection["item"].isin(drugs[drugs["ordercategoryid"].isin([15])]["item"].unique())]
a3=pd.concat([a2,second_injection])

filtered_df = a3.groupby('admissionid').filter(lambda x: len(x['item'].unique()) == 1)
filtered_df=filtered_df.groupby("item")
filtered_df.nunique().sort_values("admissionid",ascending=False)

# Ceftotaxim 1051, Ceftriaxone 370, 
# Ceftriaxon 117, Amoxicillin 71, Cefotaxim 46

#Cefotaxim (Claforan)                           1040  ...             1
#Ceftriaxon (Rocephin)                           363  ...             1
#Amoxicilline/Clavulaanzuur (Augmentin)           82  ...             1
#Cefazoline (Kefzol)                              37  ...             1
#Levofloxacine (Tavanic)                          12  ...             1

three=a[a['item'].isin(['Vancomycine','Ceftriaxon (Rocephin)','Piperacilline (Pipcil)'])]

threepat = three["admissionid"].unique()
#3967
all_drugs_threepat=a[[i in threepat for i in a["admissionid"]]]
all_drugs_threepat=all_drugs_threepat[all_drugs_threepat['start']>=0]

notinvasive_three = drugs[(drugs["ordercategoryid"]==21) & drugs["admissionid"].isin(threepat) & (drugs['start']>=0) & ~drugs['item'].isin(['Vancomycine','Ceftriaxon (Rocephin)','Piperacilline'])]
notinvasive_three = notinvasive_three[['SDD' not in i for i in notinvasive_three["item"]]]

second_injection =  drugs[(drugs["ordercategoryid"]==65) & drugs["admissionid"].isin(threepat) & (drugs['start']>=0) & ~drugs['item'].isin(['Vancomycine','Ceftriaxon (Rocephin)','Piperacilline'])]
list_allabs = set(all_drugs_threepat["item"].unique())#.union(set(notinvasive_three["item"].unique()))
second_injection = second_injection[second_injection["item"].isin(list_allabs)]


# Removing patients, parallel treated with further antibiotics
onlythree = []
for i in threepat:
    all_drugs_threepat_i=all_drugs_threepat[all_drugs_threepat["admissionid"]==i]
    all_drugs_threepat_ab_i = all_drugs_threepat_i[[i in ['Vancomycine','Ceftriaxon (Rocephin)','Piperacilline (Pipcil)'] for i in all_drugs_threepat_i['item']]]
    #time_min = min(all_drugs_threepat_ab_i["start"])
    #time_max = max(all_drugs_threepat_ab_i["stop"])
    
    all_drugs_threepat_further_ab_i = all_drugs_threepat_i[[i not in ['Vancomycine','Ceftriaxon (Rocephin)','Piperacilline (Pipcil)'] for i in all_drugs_threepat_i['item']]]
    #all_drugs_threepat_further_ab_i = all_drugs_threepat_further_ab_i[((all_drugs_threepat_further_ab_i["start"]>time_min) & (all_drugs_threepat_further_ab_i["start"]>time_max)) | ((all_drugs_threepat_further_ab_i["stop"]<time_max) & (all_drugs_threepat_further_ab_i["stop"]>time_min))]
    
    notinvasive_three_i = notinvasive_three[notinvasive_three["admissionid"]==i]
    #notinvasive_three_i = notinvasive_three_i[((notinvasive_three_i["start"]>time_min) & (notinvasive_three_i["start"]>time_max)) | ((notinvasive_three_i["stop"]<time_max) & (notinvasive_three_i["stop"]>time_min))]
    
    second_injection_i = second_injection[second_injection["admissionid"]==i]
    #second_injection_i = second_injection_i[((second_injection_i["start"]>time_min) & (second_injection_i["start"]>time_max)) | ((second_injection_i["stop"]<time_max) & (second_injection_i["stop"]>time_min))]
         
    if len(all_drugs_threepat_further_ab_i)==0 and len(notinvasive_three_i)==0 and len(second_injection_i)==0:
        onlythree.append(i)


# 369 Patients!!!
pat_onlythree = a[a["admissionid"].isin(onlythree)]

pat_onlythree.to_csv('/work/wendland/Amsterdamdata/drugs_threeab.csv')

with open('/work/wendland/Amsterdamdata/drug_threeab.pkl', 'wb') as handle:
    pickle.dump(pat_onlythree, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/patient_ids_threeab.pkl', 'wb') as handle:
    pickle.dump(onlythree, handle, protocol=pickle.HIGHEST_PROTOCOL)


len(pat_onlythree[pat_onlythree["item"]=='Vancomycine']["admissionid"].unique()) #Vancomycin 35
len(pat_onlythree[pat_onlythree["item"]=='Ceftriaxon (Rocephin)']["admissionid"].unique()) #Ceftriaxon 365
len(pat_onlythree[pat_onlythree["item"]=='Piperacilline (Pipcil)']["admissionid"].unique()) #0 Piperacillin


#gender, agegroup, weightgroup, heightgroup
patient_df = admissions[admissions["admissionid"].isin(onlythree)]
patient_df = patient_df[["admissionid","gender","agegroup","weightgroup","heightgroup"]]
admission_df=patient_df

with open('/work/wendland/Amsterdamdata/admission_data.pkl', 'wb') as handle:
    pickle.dump(patient_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

#### Creating smaller csv datasets and extracting patient data
chunk_size = 10000000  # Adjust this according to your needs
input_file = '/work/wendland/AmsterdamUMCdb-v1.0.2/numericitems.csv'

chunks = pd.read_csv(input_file, chunksize=chunk_size, encoding='latin-1')
for i, chunk in enumerate(chunks):
    chunk.to_csv(f'/work/wendland/AmsterdamUMCdb-v1.0.2/numericitems_{i + 1}.csv', index=False)
    

numerics = pd.read_csv('/work/wendland/AmsterdamUMCdb-v1.0.2/numericitems_1.csv', encoding='latin-1')
patient_df = numerics[numerics["admissionid"].isin(onlythree)]

for i in range(2,100):
    numerics = pd.read_csv(f'/work/wendland/AmsterdamUMCdb-v1.0.2/numericitems_{i}.csv', encoding='latin-1')
    patient_df = pd.concat([patient_df,numerics[numerics["admissionid"].isin(onlythree)]])


patient_df.to_csv('/work/wendland/Amsterdamdata/numerics_sepsis_patients.csv')

patient_df=pd.read_csv('/work/wendland/Amsterdamdata/numerics_sepsis_patients.csv',encoding='latin-1')

with open('/work/wendland/Amsterdamdata/numerics_sepsis_patients.pkl', 'wb') as handle:
    pickle.dump(patient_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

listitems = pd.read_csv('/work/wendland/AmsterdamUMCdb-v1.0.2/listitems.csv', encoding='latin-1')

listitems = listitems[listitems["admissionid"].isin(onlythree)]

listitems["item"].unique()

with open('/work/wendland/Amsterdamdata/listitems_patients.pkl', 'wb') as handle:
    pickle.dump(listitems, handle, protocol=pickle.HIGHEST_PROTOCOL)

#### Preprocessing laboratory values


#SOFA-Score, alanine transaminase (in IU/L), anion gap (in mEq/L), bicarbonate (in mEq/L), bilirubin total (in mg/dl), blood urea nitrogen (in mg/dl), creatinine (in mg/dl), diastolic blood pressure (in mmHg), number of platelets (in k/uL), red cell distribution width (in \%) and systolic blood pressure (in mmHg) and the selected static features are biological sex, age, height and weight at submission. 
numerics_sepsis_patients = patient_df

numerics_sepsis_patients.loc[[i == '-' for i in numerics_sepsis_patients["tag"]],'value']=-1*numerics_sepsis_patients.loc[[i == '-' for i in numerics_sepsis_patients["tag"]],'value']

lab_df_amsterdam=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ALAT' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]
lab_df_amsterdam.loc[['ALAT' in i for i in lab_df_amsterdam["item"]],'item']='ALAT (blood)'
lab_df_amsterdam.loc[['ALAT' in i for i in lab_df_amsterdam["item"]],'unit']='IU/L'

lab_df_amsterdam = pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['Anio' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Anion-Gap (bloed)' in i for i in lab_df_amsterdam["item"]],'item']='Anion-Gap (blood)'


lab_df_amsterdam = pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['HCO3' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['HCO3' in i for i in lab_df_amsterdam["item"]],'item']='Bicarbonate (blood)'
lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'Bicarbonate (blood)') & ((lab_df_amsterdam['value'] < 0) | (lab_df_amsterdam['value'] > 100)))]

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Bilirubine (bloed)', 'Bili Totaal'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'item']='Bilirubin total'
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'value']/17.1,decimals=1)
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Ureum (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Ureum' in i for i in lab_df_amsterdam["item"]],'item']='BUN'
lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'value']/0.3571,decimals=1)
lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Kreatinine (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Krea' in i for i in lab_df_amsterdam["item"]],'item']='Creatinine'
lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'value']/88.42,decimals=1)
lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ABP dias' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP diastolisch II','Niet invasieve bloeddruk diastolisch'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['IABP Mean Dia. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['ABP diastolisch','ABP diastolisch II','Niet invasieve bloeddruk diastolisch','IABP Mean Dia. Blood Pressure'] for i in lab_df_amsterdam["item"]],'item']='ABP diastolic'
lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'ABP diastolic') & (lab_df_amsterdam['value'] <= 0))]

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ABP systolisch' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP systolisch II','Niet invasieve bloeddruk systolisch'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['IABP Mean Sys. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['ABP systolisch','ABP systolisch II','Niet invasieve bloeddruk systolisch','IABP Mean Sys. Blood Pressure'] for i in lab_df_amsterdam["item"]],'item']='ABP systolic'


lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'ABP diastolic') & ((lab_df_amsterdam['value'] <= 0) | (lab_df_amsterdam['value'] > 500)))]
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ["Thrombo's (bloed)"] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ["Thrombo's (bloed)", 'Thrombocyten'] for i in lab_df_amsterdam["item"]],'item']='Platelets (blood)'
lab_df_amsterdam.loc[['Platelets (blood)' in i for i in lab_df_amsterdam["item"]],'unit'] = 'k/mul'

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['RDW (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['RDW (bloed)'] for i in lab_df_amsterdam["item"]],'item']='RDW (blood)'
lab_df_amsterdam = lab_df_amsterdam[['admissionid','item','value','unit','measuredat']]

lab_df_amsterdam['measuredat']=np.round(lab_df_amsterdam['measuredat']/3600000)

lab_df_amsterdam.to_csv('/work/wendland/Amsterdamdata/lab_amsterdam_mimicvars.csv')


##### Preprocessing with utils script

import utils_paper as u


sofa_score = pd.read_csv('/work/wendland/Amsterdamdata/sofa_score_hourly.csv')

with open('/work/wendland/data_prep_three/sepsis_all_1_variables_mean.pkl', 'rb') as handle:
    variables_mean_pre = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_variables_std.pkl', 'rb') as handle:
    variables_std_pre = pickle.load(handle)

with open('/work/wendland/data_prep_three/sepsis_all_1_variables_complete.pkl', 'rb') as handle:
    variables_complete_mimic = pickle.load(handle)

with open('/work/wendland/amsterdamdata/new_version_knnimputer.pkl', 'rb') as handle:
    imputer = pickle.load(handle)
    
with open('/work/wendland/Amsterdamdata/fit_keys.pkl', 'rb') as handle:
    fit_keys_mimic = pickle.load(handle)

with open('/work/wendland/Amsterdamdata/lab_amsterdam_mimicvars.pkl', 'rb') as handle:
    lab_df_amsterdam = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_static_mean.pkl', 'rb') as handle:
    static_mean_pre = pickle.load(handle)
    
with open('/work/wendland/data_prep_three/sepsis_all_1_static_std.pkl', 'rb') as handle:
    static_std_pre = pickle.load(handle)
    

variables_amsterdam = ['SOFA','Anion-Gap (blood)','Bicarbonate (blood)','BUN','Creatinine','ABP diastolic', 'Platelets (blood)', 'RDW (blood)', 'ABP systolic', 'Bilirubin total', 'ALAT (blood)']

lab_df=lab_df_amsterdam

lab_df = lab_df.rename(columns={"item": 'label',"admissionid": 'hadm_id'})    
pat_onlythree2 = pat_onlythree[['admissionid','item','dose','doseunit','start']]
pat_onlythree2 = pat_onlythree2.rename(columns={"item": 'label',"admissionid": 'hadm_id'})
pat_onlythree2['start'] = np.round(pat_onlythree2['start']/3600000)
antibiotics_variables=['Vancomycine', 'Piperacilline (Pipcil)','Ceftriaxon (Rocephin)']

min_pred=0
max_pred=None
pred_times_in_h=1
thresh = None # or thresh=0.5
sep_data=sepsis3
round_time=False
round_nearest=True #round to hours
round_minutes=60    
aggregate_startvalues = True
start_hours = 1 # hours to aggregate the startvalues
remove_unaggregated_values=True
timetype = 'measuredat'
start_timepoint='sepsis_icu'
missing_mask=True
lab_demo=None # or admission_data???
list_of_hadms=onlythree #or sepsisids
first_adms=None
variables=variables_amsterdam # or list of features
print_=True
just_icu=None
icustays=None 
stays_list=onlythree 
missing_imputation_start=True
antibiotics=pat_onlythree2 
remove_noant=True
static=patient_df
static_time=False
standardize=True 
train_test_split=False
seed=0
antibiotics_variables=['Vancomycine', 'Piperacilline (Pipcil)','Ceftriaxon (Rocephin)']
binary_antibiotics=True
static_bin_ant=True

cut = 'antibiotics'# 'Sepsis'
variables_mean_pre=variables_mean_pre #None
variables_std_pre=variables_std_pre #None
#static_mean_pre=static_mean_pre[:3] #None
#static_std_pre=static_std_pre[:3] #None
fit = imputer
dataset = 'amsterdamumcdb' or 'mimic-iv'
sofa_amsterdam = sofa_score

admission_data=patient_df

tensor, key_dict, variables_tensor, variables_complete, complete_stay_list, static_tensor, static_variables, fit, variables_mean, variables_std, static_mean, static_std, indices_train, indices_test=u.multiple_patients_predictions_tensor(min_pred=0,max_pred=None, pred_times_in_h=1,lab_df=lab_df,thresh=None, sep_data=sepsis3, round_time=False, round_nearest=True, round_minutes=60, aggregate_startvalues=True, start_hours=1, remove_unaggregated_values=True, start_timepoint='sepsis_icu', lab_demo=None,print_=True,list_of_hadms=onlythree, just_icu=None,icustays=None,stays_list=onlythree, variables=variables_amsterdam, missing_imputation_start=True, antibiotics=pat_onlythree2,remove_noant=True, static=admission_data, static_time=False, standardize=True, train_test_split=False, seed=0,antibiotics_variables=antibiotics_variables,binary_antibiotics=True,static_bin_ant=True, cut = 'antibiotics', variables_mean_pre=variables_mean_pre, variables_std_pre=variables_std_pre, static_mean_pre=static_mean_pre, static_std_pre=static_std_pre, fit = imputer, dataset = 'amsterdamumcdb', sofa_amsterdam = sofa_score, timetype=timetype)

# tensor: Tensor containing all dynamic variables (including time on the first channel, antibiotics, and missing masks) for OptAB, size: batch x timepoints x variables
# key_dict: Dictionary mapping time index to time in hours (here 1 index = 1 hour, therefore not necessary)
# variables_tensor: List containing all labels of the dynamic variables, which are standardized (variables, without time and missing masks)
# variables_complete: List containing all labels of the dynamic variables
# complete_stay_list: List of stay indices for the patients
# static_tensor: Tensor containing all static variables for OptAB, size: batch x variables
# static_variables: List containing all labels of the static variables
# fit: Fitted knn_imputer object 
# variables_mean and variables_std: Lists containing all means and standard deviations used for standardization of the dynamic variables
# static_mean and static_std: Lists containing all means and standard deviations used for standardization of the static variables
# indices_train and indices_test: Lists containing the training and test indices of the data


with open('/work/wendland/Amsterdamdata/sepsis_all_1_lab_amsterdam.pkl', 'wb') as handle:
    pickle.dump(tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_keys_amsterdam.pkl', 'wb') as handle:
    pickle.dump(key_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_variables_amsterdam.pkl', 'wb') as handle:
    pickle.dump(variables_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_variables_complete_amsterdam.pkl', 'wb') as handle:
    pickle.dump(variables_complete, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_complete_stays_amsterdam.pkl', 'wb') as handle:
    pickle.dump(complete_stay_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_static_amsterdam.pkl', 'wb') as handle:
    pickle.dump(static_tensor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_static_variables_amsterdam.pkl', 'wb') as handle:
    pickle.dump(static_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_imputer_amsterdam.pkl', 'wb') as handle:
    pickle.dump(fit, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_variables_mean_amsterdam.pkl', 'wb') as handle:
    pickle.dump(variables_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_variables_std_amsterdam.pkl', 'wb') as handle:
    pickle.dump(variables_std, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_static_mean_amsterdam.pkl', 'wb') as handle:
    pickle.dump(static_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_static_std_amsterdam.pkl', 'wb') as handle:
    pickle.dump(static_std, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_indices_train_amsterdam.pkl', 'wb') as handle:
    pickle.dump(indices_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/work/wendland/Amsterdamdata/sepsis_all_1_indices_test_amsterdam.pkl', 'wb') as handle:
    pickle.dump(indices_test, handle, protocol=pickle.HIGHEST_PROTOCOL)



-
