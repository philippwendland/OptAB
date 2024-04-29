import pandas as pd

import pickle
import numpy as np

with open('/work/wendland/Amsterdamdata/drug_threeab.pkl', 'rb') as handle:
    pat_onlythree = pickle.load(handle)

with open('/work/wendland/Amsterdamdata/patient_ids_threeab.pkl', 'rb') as handle:
    onlythree = pickle.load(handle)

import matplotlib.pyplot as plt
import seaborn as sns

# Creating plots and read variable names etc. to preprocesse the amsterdamumcdb data
# Conversion of units

numerics_sepsis_patients = pd.read_csv('/work/wendland/Amsterdamdata/numerics_sepsis_patients.csv', encoding='latin-1')


numerics_sepsis_patients.loc[[i == '-' for i in numerics_sepsis_patients["tag"]],'value']=-1*numerics_sepsis_patients.loc[[i == '-' for i in numerics_sepsis_patients["tag"]],'value']
# Negative base excess has often "-" as tag


#SOFA-Score, alanine transaminase (in IU/L), anion gap (in mEq/L), bicarbonate (in mEq/L), bilirubin total (in mg/dl), blood urea nitrogen (in mg/dl), creatinine (in mg/dl), diastolic blood pressure (in mmHg), number of platelets (in k/uL), red cell distribution width (in \%) and systolic blood pressure (in mmHg) and the selected static features are biological sex, age, height and weight at submission. 

numerics_sepsis_patients[['ALAT' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'ALAT (bloed)', 'ALAT'
#E/L = eenheid/liter same as IU/L, https://unitslab.com/de/node/30

lab_df_amsterdam=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ALAT' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]
lab_df_amsterdam.loc[['ALAT' in i for i in lab_df_amsterdam["item"]],'item']='ALAT (blood)'
lab_df_amsterdam.loc[['ALAT' in i for i in lab_df_amsterdam["item"]],'unit']='IU/L'

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ALAT' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359

min(a["value"])
max(a["value"])
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ALAT complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('IU/L')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ALAT complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('IU/L')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ALAT' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ALAT (bloed)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')



numerics_sepsis_patients[['Anio' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'Anion-Gap (bloed)'
#mmol/l, same as mEq/l

lab_df_amsterdam = pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['Anio' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Anion-Gap (bloed)' in i for i in lab_df_amsterdam["item"]],'item']='Anion-Gap (blood)'

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['Anio' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]

len(a["admissionid"].unique()) # 293

min(a["value"])
max(a["value"])
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


numerics_sepsis_patients[['HCO3' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'Act.HCO3 (bloed)', 'HCO3'
#mmol/l, same as mEq/l

lab_df_amsterdam = pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['HCO3' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
#b=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['Act.HCO3' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]
lab_df_amsterdam.loc[['HCO3' in i for i in lab_df_amsterdam["item"]],'item']='Bicarbonate (blood)'

lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'Bicarbonate (blood)') & ((lab_df_amsterdam['value'] < 0) | (lab_df_amsterdam['value'] > 100)))]

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['HCO3' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 368

min(a["value"])
max(a["value"])

a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('HCO3 complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('HCO3 complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Act.HCO3 (bloed)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'HCO3' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')



numerics_sepsis_patients[['Bili' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'Bilirubine (bloed)', 'Gecon.Bili (bloed)','Bilirubine geconjugeerd', 'Bili Totaal','Bilirubine drainvocht (drain)', 'Gecon.Bili  (bloed)'
# gecon 'direct or conjugated'
# -> 'Bilirubine (bloed)','Bili Totaal' (3 times)
# Âµmol/l is µmol/l 1 mg/dl = 17.1 µmol/l
numerics_sepsis_patients[['BILI' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'BILIRUBINE (overig)' not interesting

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Bilirubine (bloed)', 'Bili Totaal'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'item']='Bilirubin total'
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'value']/17.1,decimals=1)
lab_df_amsterdam.loc[['Bili' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Bilirubine (bloed)', 'Bili Totaal'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 338

min(a["value"])
max(a["value"])
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Bilirubin complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('µmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Bilirubin complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('µmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Bilirubine (bloed)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Bili Totaal' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')



#len(a["admissionid"].unique()) # 338

numerics_sepsis_patients[['Ureum' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
# https://en.wikipedia.org/wiki/Blood_urea_nitrogen
# 'Ureum (bloed)', 'Ureum (urine)', 'Ureum'
# in mmol/l

# Uream maybe mixture of urine and blood....

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Ureum (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
#b=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Ureum'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
lab_df_amsterdam.loc[['Ureum' in i for i in lab_df_amsterdam["item"]],'item']='BUN'
lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'value']/0.3571,decimals=1)
lab_df_amsterdam.loc[['BUN' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Ureum (bloed)', 'Ureum'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"])
max(a["value"])
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Urea complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Urea complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Ureum (bloed)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Ureum' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Ureum (urine)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


#len(a["admissionid"].unique()) # 366

numerics_sepsis_patients[['Krea' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'Kreatinine (bloed)', 'Kreatinine (urine)',
#       'Kreatinine (verz. urine)', 'Kreatinine',
#       'Kreatinine  (verz. urine)', 'A_Serum_Kreatinine',
#       'Kreat.klaring (urine)', 'Kreatinine (overig)'
numerics_sepsis_patients[['KREA' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'KREAT.KLAR (verz. urine)', 'KREAT enzym. (bloed)', not interesting

# interesting: 'Kreatinine (bloed)','Kreatinine','A_Serum_Kreatinine','MCA_Serum_Kreatinine', 1 mg/dl = 88.42 mmol/l -> /88.42
#'Kreatinine' unklar, MCE_Serum_Kreatinine

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Kreatinine (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[['Krea' in i for i in lab_df_amsterdam["item"]],'item']='Creatinine'
lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'value']=np.round(lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'value']/88.42,decimals=1)
lab_df_amsterdam.loc[['Creatinine' in i for i in lab_df_amsterdam["item"]],'unit'] = 'mg/dl'


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Kreatinine (bloed)','Kreatinine','A_Serum_Kreatinine','MCA_Serum_Kreatinine'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"]) #19
max(a["value"]) #2157
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Creatinine complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Creatinine complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('mmol/l')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Kreatinine (bloed)' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Kreatinine' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'A_Serum_Kreatinine' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'MCA_Serum_Kreatinine' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')



#len(a["admissionid"].unique()) # 369

numerics_sepsis_patients[['dias' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
# unit correct
#'ABP diastolisch', 'PAP diastolisch', 
#       'Niet invasieve bloeddruk diastolisch',
#       'PiCCO GEDVI (Globaal Einddiastolisch Volume Index)',
#       'ABP diastolisch II'

#PAP different

#numerics_sepsis_patients[['DIAS' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#'Global End Diastolic Volume Index' not interesting

numerics_sepsis_patients[['BP' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#array(['ABP systolisch', 'ABP gemiddeld', 'ABP diastolisch',
#       'Streef Onderwaarde ABPm', 'BPS score', 'Streef Bovenwaarde ABPm',
#       'Streef Bovenwaarde ABPs', 'Streef Onderwaarde ABPs',
#       'IABP Systole', 'IABP Augmentatie', 'IABP HF',
#       'IABP Mean Sys. Blood Pressure', 'IABP Mean Dia. Blood Pressure',
#       'IABP Mean Blood Pressure', 'A_ABPm', 'ABP systolisch II',
#       'ABP diastolisch II', 'ABP gemiddeld II'], dtype=object)

# ABP arterieller Blutdruck interessant,
# nicht invasiver Bludruck ist irgendwie komisch (höufig tendenziell leicht größere Werte)
# hmmm IABP Mena Dia Blood pressure ist tatsächlich auch irgendwie anders, eher mitnehmen.


lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ABP dias' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP diastolisch II','Niet invasieve bloeddruk diastolisch'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['IABP Mean Dia. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['ABP diastolisch','ABP diastolisch II','Niet invasieve bloeddruk diastolisch','IABP Mean Dia. Blood Pressure'] for i in lab_df_amsterdam["item"]],'item']='ABP diastolic'

lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'ABP diastolic') & (lab_df_amsterdam['value'] <= 0))]


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP diastolisch','ABP diastolisch II','Niet invasieve bloeddruk diastolisch','IABP Mean Dia. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"]) #19
max(a["value"]) #2157
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ABP diastolic complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('mmHg')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ABP diastolic complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('mmHg')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ABP diastolisch' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ABP diastolisch II' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Niet invasieve bloeddruk diastolisch' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'IABP Mean Dia. Blood Pressure' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


#d = a[a["admissionid"].isin(c["admissionid"].unique())]

#len(a["admissionid"].unique()) # 361

numerics_sepsis_patients[['sys' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()

numerics_sepsis_patients[['Sys' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
# equivalent zu diastolisch


lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[['ABP systolisch' in i for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP systolisch II','Niet invasieve bloeddruk systolisch'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['IABP Mean Sys. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['ABP systolisch','ABP systolisch II','Niet invasieve bloeddruk systolisch','IABP Mean Sys. Blood Pressure'] for i in lab_df_amsterdam["item"]],'item']='ABP systolic'


lab_df_amsterdam = lab_df_amsterdam[~((lab_df_amsterdam['item'] == 'ABP diastolic') & ((lab_df_amsterdam['value'] <= 0) | (lab_df_amsterdam['value'] > 500)))]


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['ABP systolisch','ABP systolisch II','Niet invasieve bloeddruk systolisch','IABP Mean Sys. Blood Pressure'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"]) #19
max(a["value"]) #2157
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ABP systolic complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('mmHg')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('ABP systolic complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('mmHg')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ABP systolisch' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'ABP systolisch II' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Niet invasieve bloeddruk systolisch' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'IABP Mean Sys. Blood Pressure' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')

#len(a["admissionid"].unique())

numerics_sepsis_patients[['Thromb' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
#"Thrombo's (bloed)", 'Thrombocyten', 'Thrombo CD61 (bloed)'
#

# 10^9/l =  k/mul!! 

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ["Thrombo's (bloed)"] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
b=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['Thrombocyten'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ["Thrombo's (bloed)"] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ["Thrombo's (bloed)", 'Thrombocyten'] for i in lab_df_amsterdam["item"]],'item']='Platelets (blood)'
lab_df_amsterdam.loc[['Platelets (blood)' in i for i in lab_df_amsterdam["item"]],'unit'] = 'k/mul'


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ["Thrombo's (bloed)", 'Thrombocyten'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"]) #19
max(a["value"]) #2157
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Platelets complete')
ax1.boxplot(a["value"])
ax1.set_ylabel('k/mul')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('Platelets complete')
sns.violinplot(y=a["value"])
ax1.set_ylabel('k/mul')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == "Thrombo's (bloed)" for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i == 'Thrombocyten' for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 359
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
sns.violinplot(y=a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title(a['item'].unique()[0])
ax1.boxplot(a["value"])
ax1.set_ylabel(a['unit'].unique()[0])

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


#len(a["admissionid"].unique()) # 369

numerics_sepsis_patients[['RDW' in i for i in numerics_sepsis_patients["item"]]]["item"].unique()
# 'RDW (bloed)'


# M.C.V one value

lab_df_amsterdam=pd.concat([lab_df_amsterdam,numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['RDW (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]])
lab_df_amsterdam.loc[[i in ['RDW (bloed)'] for i in lab_df_amsterdam["item"]],'item']='RDW (blood)'

lab_df_amsterdam = lab_df_amsterdam[['admissionid','item','value','unit','measuredat']]


a=numerics_sepsis_patients[numerics_sepsis_patients["item"].isin(numerics_sepsis_patients[[i in ['RDW (bloed)'] for i in numerics_sepsis_patients["item"]]]["item"].unique())]
len(a["admissionid"].unique()) # 366

min(a["value"]) #19
max(a["value"]) #2157
a["value"].describe()

fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('RDW')
ax1.boxplot(a["value"])
#ax1.set_ylabel('k/mul')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/boxplot_'+ax1.get_title()+'.png')


fig1,ax1 = plt.subplots(figsize=[6,5],dpi=300)
ax1.set_title('RDW')
sns.violinplot(y=a["value"])
#ax1.set_ylabel('k/mul')

plt.savefig('/work/wendland/GitHub/TE-CDE-main/Amsterdam_boxplots/violinplot_'+ax1.get_title()+'.png')




lab_df_amsterdam['measuredat']=np.round(lab_df_amsterdam['measuredat']/3600000)

numerics_sepsis_patients["tag"].unique()
a=numerics_sepsis_patients[numerics_sepsis_patients["tag"]==numerics_sepsis_patients["tag"].unique()[8]]
#Out[8]: array([' ', 'NUL', '-', '<', 'T', '>', 'D', '0', '&', '5'], dtype=object)
#- bezieht sich auf Basenexcess und muss dann negativ berechnet werden
#ignoring rest

b=numerics_sepsis_patients[numerics_sepsis_patients["itemid"]==10053]

lab_df_amsterdam["comment"].unique()
#array(['NUL', ' ', '<', '>'], dtype=object)


numerics_sepsis_patients["comment"].unique()
a=numerics_sepsis_patients[numerics_sepsis_patients["comment"].isin(numerics_sepsis_patients["comment"].unique()[3:])]

numerics_sepsis_patients["islabresult"].unique()
a=numerics_sepsis_patients[numerics_sepsis_patients["islabresult"]==0]

with open('/work/wendland/Amsterdamdata/lab_amsterdam_mimicvars.pkl', 'wb') as handle:
    pickle.dump(lab_df_amsterdam, handle, protocol=pickle.HIGHEST_PROTOCOL)

lab_df_amsterdam.to_csv('/work/wendland/Amsterdamdata/lab_amsterdam_mimicvars.csv')













