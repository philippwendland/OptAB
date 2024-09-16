import pandas as pd
import numpy as np
import torch
import time


def sql_queries_start():
    ### SQL QUREIES TO GENERATE THE MIMINIMAL SEPSIS DATA MODEL AND MIMIC GENERATED SOFA, SOI, SOFA, AND SEPSIS DETERMINATIONS TO CHECK AGAINST
    
    adt_sql ='''WITH adt AS (SELECT
        all_adt.subject_id, -- col_id
        all_adt.hadm_id, -- adt_id
        all_adt.admittime, -- adt_admit 
        all_adt.dischtime, -- adt_discharge
        (admittime - dischtime) as LOS
    FROM mimiciv_hosp.admissions as all_adt
    INNER JOIN mimiciv_derived.icustay_times as icu
    ON all_adt.hadm_id = icu.hadm_id)
    SELECT * FROM adt'''
    
    
    icu_adt = '''WITH icu_adt AS(SELECT *
                FROM mimiciv_derived.icustay_times)
                SELECT * FROM icu_adt'''
    
    abxdf_sql = '''SELECT * FROM mimiciv_derived.antibiotic'''
    
    cxdf_sql = '''WITH cx AS (select micro_specimen_id
        -- the following columns are identical for all rows of the same micro_specimen_id
        -- these aggregates simply collapse duplicates down to 1 row
        , MAX(subject_id) AS subject_id
        , MAX(hadm_id) AS hadm_id
        , CAST(MAX(chartdate) AS DATE) AS chartdate
        , MAX(charttime) AS charttime
        , MAX(spec_type_desc) AS spec_type_desc
        , max(case when org_name is not null and org_name != '' then 1 else 0 end) as PositiveCulture
      from mimiciv_hosp.microbiologyevents
      group by micro_specimen_id)
      SELECT * FROM cx'''
    
    demodf_sql = """SELECT subject_id, hadm_id, stay_id,admittime, dischtime, los_hospital,los_icu, admission_age,gender,race,hospital_expire_flag,dod
    FROM mimiciv_derived.icustay_detail"""
    
    
    mvdf_sql = '''WITH mv AS (SELECT
        icustay_detail.subject_id,
        mimiciv_derived.ventilation.stay_id,
        mimiciv_derived.ventilation.starttime as vent_start,
        mimiciv_derived.ventilation.endtime as vent_end,
        mimiciv_derived.ventilation.ventilation_status
    FROM mimiciv_derived.ventilation -- some stays include multiple ventilation events
    JOIN mimiciv_derived.icustay_detail
        ON mimiciv_derived.ventilation.stay_id = mimiciv_derived.icustay_detail.stay_id)
        SELECT * FROM mv'''
    
    dxdf_sql = '''WITH dx AS (SELECT
        mimiciv_hosp.admissions.subject_id, -- Subject ID
        mimiciv_hosp.diagnoses_icd.hadm_id,  -- adt_id
        mimiciv_hosp.diagnoses_icd.icd_code, -- ICD code number
        mimiciv_hosp.diagnoses_icd.icd_version, --ICD version number
        mimiciv_hosp.admissions.dischtime -- dx_time (note that this is discharge time and not diagnosis time)
    FROM mimiciv_hosp.diagnoses_icd
    JOIN mimiciv_hosp.admissions
        ON mimiciv_hosp.diagnoses_icd.hadm_id = mimiciv_hosp.admissions.hadm_id)
        SELECT * FROM dx'''
    
    vasodf_sql = '''
    WITH pressors AS (WITH drugs AS((
    -- This query extracts dose+durations of phenylephrine administration
    select
      stay_id, linkorderid
      , rate as rate
      , rateuom as rate_uom
      , amount as total_dose
      , starttime
      , endtime
      ,'phenylephrine' as drug
    from mimiciv_icu.inputevents
    where itemid = 221749 -- phenylephrine
    	)
    	
    UNION
    (
    -- This query extracts dose+durations of dopamine administration
    select
    stay_id, linkorderid
    , rate as rate
    , rateuom as rate_uom
    , amount as total_dose
    , starttime
    , endtime
    ,'dobutamine' as drug
    from mimiciv_icu.inputevents
    where itemid = 221653 -- dobutamine
    	)
    
    UNION ( 
    -- This query extracts dose+durations of dopamine administration
    select
    stay_id, linkorderid
    , rate as rate
    , rateuom as rate_uom
    , amount as total_dose
    , starttime
    , endtime
    , 'dopamine' as drug
    from mimiciv_icu.inputevents
    where itemid = 221662 -- dopamine
    )
    
    UNION ( 
    -- This query extracts dose+durations of norepinephrine administration
    select
      stay_id, linkorderid
      , rate as rate
      , rateuom as rate_uom
      , amount as total_dose
      , starttime
      , endtime
      ,'norepinephrine' as drug
    from mimiciv_icu.inputevents
    where itemid = 221906 -- norepinephrine
    )
    	
    UNION (
    -- This query extracts dose+durations of epinephrine administration
    select
    stay_id, linkorderid
    , rate as rate
    , rateuom as rate_uom
    , amount as total_dose
    , starttime
    , endtime
    , 'epinephrine' as drug
    from mimiciv_icu.inputevents
    where itemid = 221289 -- epinephrine
    )
    UNION (
    -- This query extracts dose+durations of vasopressin administration
    select
      stay_id, linkorderid
      , rate as rate
      , rateuom as rate_uom
      , amount as total_dose
      , starttime
      , endtime
    , 'vasopressin' as drug
    from mimiciv_icu.inputevents
    where itemid = 222315 -- vasopressin)
    )),
    cohort AS (
      select stay_id, hadm_id, subject_id
      from mimiciv_icu.icustays
    )
    (SELECT * FROM drugs LEFT JOIN cohort ON cohort.stay_id = drugs.stay_id))
    
    SELECT subject_id,hadm_id,rate,rate_uom,starttime as vaso_start,endtime as vaso_end, drug FROM pressors
    '''
    
    
    
    lvdf_sql = '''WITH lvdf AS ((
    SELECT subject_id,charttime as time,bicarbonate as val, 'bicarb' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE bicarbonate IS NOT NULL
    )
    UNION
    (
    SELECT subject_id, charttime as time, bun as val, 'bun' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE bun IS NOT NULL
    )
    UNION
    (
    SELECT subject_id, charttime as time,creatinine as val, 'cr' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE creatinine IS NOT NULL
    )
    UNION
    (
    SELECT subject_id ,charttime as time,wbc as val, 'wbc' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE wbc IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,po2 as val, 'pO2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE po2 IS NOT NULL and specimen = 'ART.'
    )
    UNION
    (
    SELECT subject_id,charttime as time,fio2 as val, 'fio2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE fio2 IS NOT NULL and specimen = 'ART.'
    )
    UNION
    (
    SELECT subject_id,charttime as time, fio2_chartevents as val, 'fio2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE fio2_chartevents IS NOT NULL and specimen = 'ART.'
    )
    UNION
    (
    SELECT subject_id, charttime as time,bilirubin_total as val, 'bili' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE bilirubin_total IS NOT NULL 
    )
    UNION
    (
    SELECT subject_id ,charttime as time,platelet as val, 'plt' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count 
    WHERE platelet IS NOT NULL 
    )
    UNION
    (
    SELECT gcs_t.subject_id, gcs_t.charttime as time, gcs_t.gcs as val, 'gcs' as mark, stay.hadm_id
    FROM mimiciv_derived.gcs as gcs_t JOIN mimiciv_derived.icustay_detail as stay
    ON gcs_t.stay_id = stay.stay_id
    WHERE gcs_t.gcs IS NOT NULL
    )
    UNION
    (
    SELECT vs.subject_id, vs.charttime as time,mbp as val, 'map' as mark, stay.hadm_id
    FROM mimiciv_derived.vitalsign as vs JOIN mimiciv_derived.icustay_detail as stay
    ON vs.stay_id = stay.stay_id
    WHERE vs.mbp IS NOT NULL
    )
    UNION
    (
    SELECT vs.subject_id,vs.charttime as time,spo2 as val, 'spo2' as mark, stay.hadm_id
    FROM mimiciv_derived.vitalsign as vs JOIN mimiciv_derived.icustay_detail as stay
    ON vs.stay_id = stay.stay_id
    WHERE vs.spo2 IS NOT NULL 
    )
    UNION
    (
    SELECT vs.subject_id, vs.charttime as time, heart_rate as val, 'hr' as mark, stay.hadm_id
    FROM mimiciv_derived.vitalsign as vs JOIN mimiciv_derived.icustay_detail as stay
    ON vs.stay_id = stay.stay_id
    WHERE vs.spo2 IS NOT NULL 
    )
    UNION
    (
    SELECT vs.subject_id, vs.charttime as time,vs.resp_rate as val, 'rr' as mark, stay.hadm_id
    FROM mimiciv_derived.vitalsign as vs JOIN mimiciv_derived.icustay_detail as stay
    ON vs.stay_id = stay.stay_id
    WHERE vs.resp_rate IS NOT NULL 
    )
    UNION
    (
    SELECT vs.subject_id, vs.charttime as time,vs.sbp as val, 'sbp' as mark, stay.hadm_id
    FROM mimiciv_derived.vitalsign as vs JOIN mimiciv_derived.icustay_detail as stay
    ON vs.stay_id = stay.stay_id
    WHERE vs.sbp IS NOT NULL 
    )
    UNION
    (
    SELECT O2.subject_id, O2.charttime as time,O2.o2_flow as val, 'O2 Flow' as mark, stay.hadm_id
    FROM mimiciv_derived.oxygen_delivery as O2 JOIN mimiciv_derived.icustay_detail as stay
    ON O2.stay_id = stay.stay_id
    WHERE O2.o2_flow IS NOT NULL 
    ))
    SELECT * FROM lvdf'''
    
    
    Sepsis3_sql = '''WITH sepsis as (SELECT sep3.*,details.hadm_id
    FROM mimiciv_derived.sepsis3 as sep3
    INNER JOIN mimiciv_derived.icustay_detail as details
    ON sep3.subject_id = details.subject_id AND sep3.stay_id = details.stay_id)
    SELECT * FROM sepsis
    '''
    
    SOI_sql = '''WITH SOI as (SELECT *
    FROM mimiciv_derived.suspicion_of_infection)
    SELECT * FROM SOI'''
    
    SOFA_sql = '''SELECT
           subject_id, sep3.stay_id, hadm_id, icu_intime, icu_outtime, starttime, endtime,
           bilirubin_max, liver, liver_24hours, 
           meanbp_min,
           rate_dobutamine, rate_dopamine, rate_epinephrine,rate_norepinephrine,
           
           cardiovascular,cardiovascular_24hours,
           
           gcs_min,cns, cns_24hours, 
           platelet_min, coagulation,coagulation_24hours,
           creatinine_max, uo_24hr, renal, renal_24hours,
           pao2fio2ratio_novent, pao2fio2ratio_vent, respiration, respiration_24hours,
           sofa_24hours   
    
    FROM mimiciv_derived.sofa AS sep3
    LEFT JOIN mimiciv_derived.icustay_detail as detail
    ON sep3.stay_id = detail.stay_id'''
    
    
    Sepsis_codes_sql = """WITH dx as (SELECT dx.subject_id, dx.hadm_id, dx.icd_code, dx.icd_version, icd_key.long_title
    FROM mimiciv_hosp.diagnoses_icd as dx
    INNER JOIN mimiciv_hosp.d_icd_diagnoses as icd_key
    ON dx.icd_code = icd_key.icd_code AND dx.icd_version = icd_key.icd_version
    WHERE dx.icd_code  IN ('A419','A409','A412','A4101','A4102','A411','A403','A414','A4150','A413','A4151','A4152','A4153','A4159','A4189','A021','A227','A267',
     'A327','A400','A401','A408','A4181','A427','A5486','B377','99591','0389','0380','03810','03811','03812','03819','0382','03840','03841','03842',
     '03843','03844','03849','0388','99592','R6520','78552','R6521'))
     SELECT * FROM dx
    """
    
    uo_sql =  """WITH UOP AS (SELECT times.subject_id,times.hadm_id,uo.charttime,
    
    CASE WHEN uo.uo_tm_24hr >= 22 AND uo.uo_tm_24hr <= 30
              THEN uo.urineoutput_24hr / uo.uo_tm_24hr * 24 END AS uo_24hr
    
    
    FROM mimiciv_derived.urine_output_rate uo
    LEFT JOIN mimiciv_derived.icustay_times times
    ON uo.stay_id = times.stay_id) SELECT subject_id, hadm_id, charttime as uo_time, uo_24hr FROM UOP"""
    
    sql_scripts = {'adt':adt_sql,'icu_adt':icu_adt, 'vasodf': vasodf_sql, 'abxdf':abxdf_sql,'cxdf':cxdf_sql, 'demo':demodf_sql, 
                   'mvdf':mvdf_sql,'dxdf':dxdf_sql,'uo':uo_sql, 'lvdf':lvdf_sql,'Sepsis3':Sepsis3_sql,'SOI':SOI_sql, 'SOFA': SOFA_sql, 
                   'ICD_codes':Sepsis_codes_sql}
    return sql_scripts

def load_microbiologyevents(Patients=None,cnx=None):
    ### SQL QUREIES TO load microbiological data
    #Patients is a list containing patient ids
    #cnx is the connection to the database
    
    mb_sql ='''WITH mb AS (SELECT
    subject_id, hadm_id, charttime, spec_itemid, spec_type_desc, storetime,
test_itemid, test_name, org_itemid, org_name, ab_itemid, ab_name,
dilution_text, dilution_comparison, dilution_value, interpretation

    FROM mimiciv_hosp.microbiologyevents as all_mb
    WHERE interpretation = 'S' or interpretation='R' or interpretation='I' or interpretation='P')
    SELECT * FROM mb'''
    
    include_subject_id = tuple(Patients["patient_id"].to_list())
    #curtime = time.time()
    print('loading {}'.format(mb_sql),end='...')
    mb = pd.read_sql_query(mb_sql +' WHERE subject_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
    #data = pd.read_sql_query(table,cnx).drop_duplicates().reset_index(drop=True)
    
    return mb

def load_microbiologyevents_all(Patients=None,cnx=None):
    ### SQL QUREIES TO GENERATE THE MIMINIMAL SEPSIS DATA MODEL AND MIMIC GENERATED SOFA, SOI, SOFA, AND SEPSIS DETERMINATIONS TO CHECK AGAINST
    
    mb_sql ='''WITH mb AS (SELECT
    subject_id, hadm_id, charttime, spec_itemid, spec_type_desc, storetime,
test_itemid, test_name, org_itemid, org_name, ab_itemid, ab_name,
dilution_text, dilution_comparison, dilution_value, interpretation

    FROM mimiciv_hosp.microbiologyevents as all_mb)
    SELECT * FROM mb'''
    
    include_subject_id = tuple(Patients["patient_id"].to_list())
    #curtime = time.time()
    print('loading {}'.format(mb_sql),end='...')
    mb = pd.read_sql_query(mb_sql +' WHERE subject_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
    #data = pd.read_sql_query(table,cnx).drop_duplicates().reset_index(drop=True)
    
    return mb

def load_static(Patients=None,cnx=None):
    ### SQL QUREIES TO load static data
    #Patients is a list containing patient ids
    #cnx is the connection to the database
    static_sql = '''WITH Static AS ((
    SELECT t1.subject_id as patient_id, t1.stay_id, t1.gender, t1.admission_age,t3.height, t2.weight 
    FROM mimiciv_derived.icustay_detail t1 
    FULL OUTER JOIN mimiciv_derived.first_day_weight t2 on t1.stay_id=t2.stay_id
    FULL OUTER JOIN mimiciv_derived.first_day_height t3 on t1.stay_id=t3.stay_id
    ))
    SELECT * FROM static'''
        
    include_subject_id = tuple(Patients["patient_id"].to_list())
    print('loading {}'.format(static_sql),end='...')
    static = pd.read_sql_query(static_sql +' WHERE patient_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
    #data = pd.read_sql_query(table,cnx).drop_duplicates().reset_index(drop=True)
    
    return static

### FUNCTION TO LOAD DATA FROM MIMIC
def load_mimic(sql_scripts=None,cnx=None,limit=None,patients=None):
    #sql scripts are the sql queries for extracting data from mimic
    #Patients is a list containing patient ids
    #limit is a limit for loading all patients
    #cnx is the connection to the database
    data ={}
    if limit != None:
        print('loading adt')
        adt = pd.read_sql_query(sql_scripts['adt'] +' LIMIT {}'.format(limit), cnx).drop_duplicates().reset_index(drop=True)
        adt['admittime'] = pd.to_datetime(adt['admittime'],utc=True)
        adt['dischtime'] = pd.to_datetime(adt['dischtime'],utc=True)
        first = adt.sort_values(by ='admittime', ascending=True).groupby(['subject_id','hadm_id']).first().reset_index()
        last = adt.sort_values(by ='dischtime',ascending = True).groupby(['subject_id','hadm_id']).last().reset_index()
        adt = pd.merge(first.drop('dischtime',axis=1),last[['subject_id','hadm_id','dischtime']], on=['subject_id','hadm_id'], how='left')
        
        data['adt'] = adt
        
        include_subject_id = tuple(adt['subject_id'].to_list())
        include_hadm_id = tuple(adt['hadm_id'].to_list())
        
        for script in sql_scripts.keys()-['adt','Sepsis3']:
            curtime = time.time()
            print('loading {}'.format(script),end='...')
            data[script] = pd.read_sql_query(sql_scripts[script] +' WHERE subject_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
            print(f'({time.time()-curtime:.1f}s)')
        
        for script in ['Sepsis3','SOFA']:
            curtime = time.time()
            print('loading {}'.format(script),end='...')
            data[script] = pd.read_sql_query(sql_scripts[script] +' WHERE hadm_id IN {}'.format(include_hadm_id), cnx).drop_duplicates().reset_index(drop=True)
            print(f'({time.time()-curtime:.1f}s)')
    
    elif patients != None:
        print('loading adt')
        adt = pd.read_sql_query(sql_scripts['adt'], cnx).drop_duplicates().reset_index(drop=True)
        adt=adt[adt["hadm_id"].isin(patients)]
        adt['admittime'] = pd.to_datetime(adt['admittime'],utc=True)
        adt['dischtime'] = pd.to_datetime(adt['dischtime'],utc=True)
        first = adt.sort_values(by ='admittime', ascending=True).groupby(['subject_id','hadm_id']).first().reset_index()
        last = adt.sort_values(by ='dischtime',ascending = True).groupby(['subject_id','hadm_id']).last().reset_index()
        adt = pd.merge(first.drop('dischtime',axis=1),last[['subject_id','hadm_id','dischtime']], on=['subject_id','hadm_id'], how='left')
        
        data['adt'] = adt
        
        include_subject_id = tuple(adt['subject_id'].to_list())
        include_hadm_id = tuple(adt['hadm_id'].to_list())
        
        for script in sql_scripts.keys()-['adt','Sepsis3']:
            curtime = time.time()
            print('loading {}'.format(script),end='...')
            data[script] = pd.read_sql_query(sql_scripts[script] +' WHERE subject_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
            print(f'({time.time()-curtime:.1f}s)')
        
        for script in ['Sepsis3','SOFA']:
            curtime = time.time()
            print('loading {}'.format(script),end='...')
            data[script] = pd.read_sql_query(sql_scripts[script] +' WHERE hadm_id IN {}'.format(include_hadm_id), cnx).drop_duplicates().reset_index(drop=True)
            print(f'({time.time()-curtime:.1f}s)')
 
    else:
        for script in sql_scripts:
            curtime = time.time()
            print('loading {}'.format(script),end='...')
            data[script] = pd.read_sql_query(sql_scripts[script],cnx).drop_duplicates().reset_index(drop=True)
            print(f'({time.time()-curtime:.1f}s)')

        data['adt']['admittime'] = pd.to_datetime(data['adt']['admittime'],utc=True)
        data['adt']['dischtime'] = pd.to_datetime(data['adt']['dischtime'],utc=True)
        first = data['adt'].sort_values(by ='admittime', ascending=True).groupby(['subject_id','hadm_id']).first().reset_index()
        last = data['adt'].sort_values(by ='dischtime',ascending = True).groupby(['subject_id','hadm_id']).last().reset_index()
        data['adt'] = pd.merge(first.drop('dischtime',axis=1),last[['subject_id','hadm_id','dischtime']], on=['subject_id','hadm_id'], how='left')

    return data

def load_dfs(sql_scripts=None,cnx=None,patients=None):
    #LOAD DATA AND CONVERT DATA TYPES 
    #CONIDER USING THE data = load_mimic(500) TO LOAD A SMALLER PORTION OF THE DATABASE TO SAVE TIME OR MEMORY
    
    #sql scripts are the sql queries for extracting data from mimic
    #Patients is a list containing patient ids
    #cnx is the connection to the database
    
    #data = load_mimic(500)
    data = load_mimic(sql_scripts,cnx,limit=None,patients=patients)
    
    print('\n')
    data['adt']['subject_id'] = data['adt']['subject_id'].dropna().astype(int).astype(str)
    data['adt']['hadm_id'] = data['adt']['hadm_id'].dropna().astype(int).astype(str)
    data['adt']['admittime'] = pd.to_datetime(data['adt']['admittime'],utc=True,errors='raise',exact=False,infer_datetime_format=True)
    data['adt']['dischtime'] = pd.to_datetime(data['adt']['dischtime'],utc=True,errors='raise',exact=False,infer_datetime_format=True)
    
    data['adt'] = data['adt'].groupby(['subject_id','hadm_id']).agg({'admittime':'min','dischtime':'max'}).reset_index()
    
    data['adt'].dropna(inplace=True)
    data['adt'].reset_index(drop=True,inplace=True)
    
    dfs = list(data.keys())
    dfs.remove('adt')
    
    for df in dfs:
        if 'subject_id' in data[df].columns:
            data[df]['subject_id'] = data[df]['subject_id'].astype(int).astype(str)
            #data[df].dropna(subset=[col_dict['col_id']],inplace=True)
            data[df].reset_index(drop=True,inplace=True)
            
        for col in ['admittime','dischtime','icu_intime','icu_outtime','time','vent_start', 'vent_end','vaso_start','vaso_end','starttime','stoptime','charttime']:
            if col in data[df].columns:
                data[df][col] = pd.to_datetime(data[df][col],utc=True)
        
                                                             
        if 'hadm_id' in data[df].columns:
            data[df]['hadm_id'] = pd.to_numeric(data[df]['hadm_id'],errors='coerce',downcast='integer').astype(str)
            data[df]['hadm_id'] = data[df]['hadm_id'].apply(lambda z: z.split('.')[0])
            data[df].reset_index(drop=True,inplace=True)      
            
            
    data['cxdf'].dropna(subset=['charttime'],inplace=True)
    data['abxdf'].dropna(subset=['starttime'],inplace=True)
    data['icu_adt'] = data['icu_adt'].dropna(subset=['intime_hr','outtime_hr'],how='all').reset_index(drop=True)
    data['uo'] = data['uo'].dropna()
    return data

# CHANGE ADT TO CORRECT FORMAT FOR MINIMAL SEPSIS DATA MODEL
def make_adt(adt,icu_adt):
    #columns of the extracted dataframe
    
    adt_trim = adt.copy()
    adt_trim['admittime'] = pd.to_datetime(adt_trim['admittime'],utc=True)
    adt_trim['dischtime'] = pd.to_datetime(adt_trim['dischtime'],utc=True)
    

    
    adt_icu_trim = icu_adt.copy()
    adt_icu_trim = adt_icu_trim[['subject_id','hadm_id','intime_hr','outtime_hr']]
    adt_icu_trim['intime_hr'] =  pd.to_datetime(adt_icu_trim['intime_hr'],utc=True)
    adt_icu_trim['outtime_hr'] = pd.to_datetime(adt_icu_trim['outtime_hr'],utc=True)
    

    

    find_last_adt = adt_trim.sort_values(by='dischtime').groupby('hadm_id')['dischtime'].last().reset_index()
    find_last_icu = adt_icu_trim.sort_values(by='outtime_hr').groupby('hadm_id')['outtime_hr'].last().reset_index()
    find_last = pd.merge(find_last_adt,find_last_icu,on='hadm_id',how='left')
    

    find_first_adt = adt_trim.sort_values(by='admittime').groupby('hadm_id')['admittime'].first().reset_index()
    find_first_icu = adt_icu_trim.sort_values(by='intime_hr').groupby('hadm_id')['intime_hr'].first().reset_index()
    find_first = pd.merge(find_first_adt,find_first_icu,on='hadm_id',how='left')
    
    find_last['last_time']= find_last[['dischtime','outtime_hr']].max(axis=1,skipna=True, numeric_only=False)
    adt_trim = pd.merge(adt_trim,find_last,on='hadm_id',how='left')
    adt_trim['dischtime'] = adt_trim['last_time']
    
    find_first['first_time'] = find_first[['admittime','intime_hr']].min(axis=1,skipna=True, numeric_only=False)
    adt_trim = pd.merge(adt_trim,find_first,on='hadm_id',how='left')
    adt_trim['admittime'] = adt_trim['first_time']
    
    
    adt_icu_trim['loc_cat'] = 'ICU'
    
    
    adt_trim = adt_trim[['subject_id','hadm_id','admittime','dischtime']].rename(columns={'admittime':'intime_hr','dischtime':'outtime_hr'})

    
    combine = pd.concat([adt_trim,adt_icu_trim])
    combine['intime_hr'] = pd.to_datetime(combine['intime_hr'],utc=True)
    combine['outtime_hr'] = pd.to_datetime(combine['outtime_hr'],utc=True)
    
    
    df = combine.groupby(['subject_id','hadm_id']).agg({'intime_hr':lambda z: list(z),'outtime_hr':lambda z: list(z)}).reset_index()
    df['all_times'] = df['intime_hr'] + df['outtime_hr']
    df = df[['subject_id','hadm_id','all_times']].explode('all_times').sort_values(by=['subject_id','hadm_id','all_times']).reset_index(drop=True)

    
    
    df['outtime_hr'] = df.groupby(['subject_id','hadm_id'])['all_times'].shift(-1)
    df = df.dropna().rename(columns={'all_times':'intime_hr'})

    df['duration'] = (df['outtime_hr'] - df['intime_hr']).apply(lambda z: z.total_seconds()/3600)
    df = df[df['duration']>0].reset_index(drop=True)
    df = pd.merge(df,adt_icu_trim[['subject_id','hadm_id','intime_hr','loc_cat']],on=['subject_id','hadm_id','intime_hr'],how='left').fillna('non-ICU')
    
    df = pd.merge(df,adt[['subject_id','hadm_id','admittime','dischtime']],on=['subject_id','hadm_id'], how='left')

    return df.dropna()

def cleanup_sep3(Sep_3=None, RTI_full=None,SOI_full=None,data=None,names_to_standard=None,cnx=None):
    #wrapper for cleaning Sepsis data of OpenSep
    #Sep_3, RTI_full and SOI_full are outcomes of OpenSep
    #data is the extracted Mimic data
    #cnx is the connection to the database
    
    Sep_3_orig = Sep_3.copy()
    RTI_orig = RTI_full.copy()
    SOI_orig = SOI_full.copy()
    
    MSep_3 = data['Sepsis3'].copy().rename(columns = names_to_standard)
    MIMIC_SOI = data['SOI'].copy().rename(columns = names_to_standard)
    MIMIC_SOFA = data['SOFA'].copy().rename(columns={'starttime':'SOFA_start', 'endtime':'SOFA_end'}).rename(columns = names_to_standard)
    Sep_3['SOITime'] = pd.to_datetime(Sep_3['SOITime'], utc=True)
    Sep_3['score_time'] = pd.to_datetime(Sep_3['score_time'], utc=True)
    Sep_3['patient_id'] = Sep_3['patient_id'].astype(str)
    Sep_3['encounter_id'] = Sep_3['encounter_id'].astype(str)
    Sep_3['loc_cat'] = Sep_3['loc_cat'].astype(str)
    Sep_3['Score'] = Sep_3['Score'].astype(str)
    
    
    Sep_3['Sepsis_Time'] = Sep_3['SOITime']
    Sep_3 = Sep_3[['patient_id', 'encounter_id', 'SOITime', 'score_time','Sepsis_Time', 'loc_cat', 'Score']]
    
    RTI = RTI_full[['patient_id', 'encounter_id', 'score_time','SOFA_GCS_Score','SOFA_MAP_Score','SOFA_BILI_Score','SOFA_PLT_Score','SOFA_RENAL_Score','SOFA_RESP_Score','SOFA_Score']].reset_index(drop=True)
    
    RTI['score_time'] = pd.to_datetime(RTI['score_time'],utc=True)
    RTI['patient_id'] = RTI['patient_id'].astype(str)
    RTI['encounter_id'] = RTI['encounter_id'].astype(str)
    python_SOFA = RTI[['patient_id','encounter_id','score_time','SOFA_GCS_Score','SOFA_MAP_Score','SOFA_BILI_Score','SOFA_PLT_Score','SOFA_RENAL_Score','SOFA_RESP_Score','SOFA_Score']] 
    
    SOI_trim = SOI_full

    SOI_trim['abx_start'] = pd.to_datetime(SOI_trim['abx_start'],utc=True)
    SOI_trim['culture_time'] = pd.to_datetime(SOI_trim['culture_time'],utc=True)
    SOI_trim['SOITime'] = pd.to_datetime(SOI_trim['SOITime'],utc=True)
    SOI_trim['patient_id'] = SOI_trim['patient_id'].astype(str)
    SOI_trim['encounter_id'] = SOI_trim['encounter_id'].astype(str)
    
    MSep_3['antibiotic_time'] = pd.to_datetime(MSep_3['antibiotic_time'],utc=True)
    MSep_3['culture_time'] = pd.to_datetime(MSep_3['culture_time'],utc=True)
    MSep_3['suspected_infection_time'] = pd.to_datetime(MSep_3['suspected_infection_time'],utc=True)
    MSep_3['sofa_time'] = pd.to_datetime(MSep_3['sofa_time'],utc=True)
    MSep_3['patient_id'] = MSep_3['patient_id'].astype(str)
    MSep_3['encounter_id'] = MSep_3['encounter_id'].astype(str)    
        
    # Find time of Sepsis for Mimic

    MSep_3['MSep_Time'] = MSep_3['suspected_infection_time']
    MSep_3 = MSep_3.sort_values(by='MSep_Time').groupby('encounter_id').first().reset_index()
    MSep_3 = MSep_3[['patient_id','encounter_id', 'stay_id', 'antibiotic_time', 'culture_time', 'suspected_infection_time', 
                     'sofa_time','MSep_Time', 'sofa_score', 'respiration', 'coagulation', 'liver', 'cardiovascular', 'cns', 'renal', 'sepsis3']]
    MIMIC_SOI['patient_id']  = MIMIC_SOI['patient_id'].astype(str)
    MIMIC_SOI['encounter_id']  = MIMIC_SOI['encounter_id'].astype(str)
    
    MIMIC_SOI['antibiotic_time'] = pd.to_datetime(MIMIC_SOI['antibiotic_time'],utc=True)
    MIMIC_SOI['suspected_infection_time'] = pd.to_datetime(MIMIC_SOI['suspected_infection_time'],utc=True)
    MIMIC_SOI['culture_time'] = pd.to_datetime(MIMIC_SOI['culture_time'],utc=True)
    
    MIMIC_SOI = MIMIC_SOI[MIMIC_SOI['suspected_infection']==1]    
    
    MIMIC_SOFA['patient_id'] = MIMIC_SOFA['patient_id'].astype(str)
    MIMIC_SOFA['encounter_id'] = MIMIC_SOFA['encounter_id'].astype(str)
    
    MIMIC_SOFA['icu_intime'] = pd.to_datetime(MIMIC_SOFA['icu_intime'],utc=True)
    MIMIC_SOFA['icu_outtime'] = pd.to_datetime(MIMIC_SOFA['icu_outtime'],utc=True)
    MIMIC_SOFA['SOFA_start'] = pd.to_datetime(MIMIC_SOFA['SOFA_start'],utc=True)
    MIMIC_SOFA['SOFA_end'] = pd.to_datetime(MIMIC_SOFA['SOFA_end'],utc=True)
    
        
    ### Remove cultures with only date if used for SOI in MIMIC
    
    raw_cxdf = pd.read_sql('''WITH cx AS (select micro_specimen_id
    -- the following columns are identical for all rows of the same micro_specimen_id
    -- these aggregates simply collapse duplicates down to 1 row
    , MAX(subject_id) AS subject_id
    , MAX(hadm_id) AS hadm_id
    , CAST(MAX(chartdate) AS DATE) AS chartdate
    , MAX(charttime) AS charttime
    , MAX(spec_type_desc) AS spec_type_desc
    , max(case when org_name is not null and org_name != '' then 1 else 0 end) as PositiveCulture
  from mimiciv_hosp.microbiologyevents
  group by micro_specimen_id)
  SELECT * FROM cx''',cnx)
    no_time = raw_cxdf[raw_cxdf['charttime'].isna()][['subject_id','chartdate','spec_type_desc']].rename(columns={'subject_id':'patient_id','chartdate':'culture_time','spec_type_desc':'specimen'}).reset_index(drop=True)
    no_time['patient_id'] = no_time['patient_id'].astype(str)
    no_time['culture_time'] = pd.to_datetime(no_time['culture_time'],utc=True)
    
    df = raw_cxdf.copy()
    df.loc[df['charttime'].isna(),'charttime'] = df.loc[df['charttime'].isna(),'chartdate']
    
    df['charttime'] = pd.to_datetime(df['charttime'],utc=True)
    df['subject_id'] = df['subject_id'].astype(str)
    
    df = df[['subject_id','charttime']].rename(columns={'subject_id':'patient_id','charttime':'culture_time'})
    
    included_cx = pd.merge(MIMIC_SOI,df,on = ['patient_id','culture_time'],how='inner')
    SOI_by_date = pd.merge(no_time,included_cx, on = ['patient_id','culture_time'], how='inner')[['patient_id','encounter_id', 'culture_time']]
    
    MSep_3 = MSep_3[~MSep_3['patient_id'].isin(SOI_by_date['patient_id'])].reset_index(drop=True)
    MIMIC_SOI = MIMIC_SOI[~MIMIC_SOI['patient_id'].isin(SOI_by_date['patient_id'])].reset_index(drop=True)

    Sep_3 = Sep_3[~Sep_3['patient_id'].isin(SOI_by_date['patient_id'])].reset_index(drop=True)
    SOI_trim = SOI_trim[~SOI_trim['patient_id'].isin(SOI_by_date['patient_id'])].reset_index(drop=True)
    
    SOI_trim = SOI_trim.sort_values(by='SOITime').groupby('encounter_id').first().reset_index() #take only first SOI
    Sep_3 = pd.merge(Sep_3,SOI_trim[['patient_id','encounter_id','SOITime']],on=['patient_id','encounter_id','SOITime'],how='inner').reset_index(drop=True)
    
    MIMIC_SOI = MIMIC_SOI.sort_values(by='suspected_infection_time').groupby('encounter_id').first().reset_index() #take only first SOI
    MSep_3 = pd.merge(MSep_3,MIMIC_SOI[['patient_id','encounter_id','suspected_infection_time']],on=['patient_id','encounter_id','suspected_infection_time'],how='inner').reset_index(drop=True)
        
    return Sep_3_orig,RTI_orig,SOI_orig,MSep_3,MIMIC_SOI,MIMIC_SOFA,Sep_3, RTI,python_SOFA,SOI_trim

def load_sepsis_dat_sql():
    #function providing sql scripts for loading Sepsis data
    bg_sql = '''WITH bg AS ((
    SELECT subject_id,charttime as time,storetime,so2 as val, 'so2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE so2 IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,po2 as val, 'po2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE po2 IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,pco2 as val, 'pco2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE pco2 IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,fio2 as val, 'fio2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE fio2 IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,fio2_chartevents as val, 'fio2_chartevents' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE fio2_chartevents IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,aado2_calc as val, 'aado2_calc' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE aado2_calc IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,pao2fio2ratio as val, 'pao2fio2ratio' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE pao2fio2ratio IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ph as val, 'ph' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE ph IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,baseexcess as val, 'baseexcess' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE baseexcess IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bicarbonate as val, 'bicarbonate' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE bicarbonate IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,totalco2 as val, 'totalco2' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE totalco2 IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,hematocrit as val, 'hematocrit' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE hematocrit IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,hemoglobin as val, 'hemoglobin' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE hemoglobin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,carboxyhemoglobin as val, 'carboxyhemoglobin' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE carboxyhemoglobin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,methemoglobin as val, 'methemoglobin' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE methemoglobin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,chloride as val, 'chloride' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE chloride IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,calcium as val, 'ionized calcium' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE calcium IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,temperature as val, 'temperature' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE temperature IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,potassium as val, 'potassium' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE potassium IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,sodium as val, 'sodium' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE sodium IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,lactate as val, 'lactate' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE lactate IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,glucose as val, 'glucose' as mark, hadm_id
    FROM mimiciv_derived.bg
    WHERE glucose IS NOT NULL
    )
    )
    
    SELECT * FROM bg'''
    
    blood_differential_sql = '''WITH blood_differential AS ((
    SELECT subject_id,charttime as time,storetime,wbc as val, 'wbc' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE wbc IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,basophils_abs as val, 'basophils_abs' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE basophils_abs IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,eosinophils_abs as val, 'eosinophils_abs' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE eosinophils_abs IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,lymphocytes_abs as val, 'lymphocytes_abs' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE lymphocytes_abs IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,monocytes_abs as val, 'monocytes_abs' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE monocytes_abs IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,neutrophils_abs as val, 'neutrophils_abs' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE neutrophils_abs IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,basophils as val, 'basophils' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE basophils IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,eosinophils as val, 'eosinophils' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE eosinophils IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,lymphocytes as val, 'lymphocytes' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE lymphocytes IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,monocytes as val, 'monocytes' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE monocytes IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,neutrophils as val, 'neutrophils' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE neutrophils IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,atypical_lymphocytes as val, 'atypical_lymphocytes' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE atypical_lymphocytes IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bands as val, 'bands' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE bands IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,immature_granulocytes as val, 'immature_granulocytes' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE immature_granulocytes IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,metamyelocytes as val, 'metamyelocytes' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE metamyelocytes IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,nrbc as val, 'nrbc' as mark, hadm_id
    FROM mimiciv_derived.blood_differential
    WHERE nrbc IS NOT NULL
    )
    )
    
    SELECT * FROM blood_differential'''
    
    chemistry_sql = '''WITH chemistry AS ((
    SELECT subject_id,charttime as time,storetime,albumin as val, 'albumin' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE albumin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,globulin as val, 'globulin' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE globulin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,total_protein as val, 'total_protein' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE total_protein IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,aniongap as val, 'aniongap' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE aniongap IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bicarbonate as val, 'bicarbonate' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE bicarbonate IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bun as val, 'bun' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE bun IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,calcium as val, 'calcium' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE calcium IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,chloride as val, 'chloride' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE chloride IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,creatinine as val, 'creatinine' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE creatinine IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,glucose as val, 'glucose' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE glucose IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,sodium as val, 'sodium' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE sodium IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,potassium as val, 'potassium' as mark, hadm_id
    FROM mimiciv_derived.chemistry
    WHERE potassium IS NOT NULL
    )
    )
    
    SELECT * FROM chemistry'''

    cbc_sql = '''WITH cbc AS ((
    SELECT subject_id,charttime as time,storetime,hematocrit as val, 'hematocrit' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE hematocrit IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,hemoglobin as val, 'hemoglobin' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE hemoglobin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,mch as val, 'mch' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE mch IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,mchc as val, 'mchc' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE mchc IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,mcv as val, 'mcv' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE mcv IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,platelet as val, 'platelet' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE platelet IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,rbc as val, 'rbc' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE rbc IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,rdw as val, 'rdw' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE rdw IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,rdwsd as val, 'rdwsd' as mark, hadm_id
    FROM mimiciv_derived.complete_blood_count
    WHERE rdwsd IS NOT NULL
    )
    
    )
    
    SELECT * FROM cbc'''
    
    enzyme_sql = '''WITH enzyme AS ((
    SELECT subject_id,charttime as time,storetime,alt as val, 'alt' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE alt IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,alp as val, 'alp' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE alp IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ast as val, 'ast' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE ast IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,amylase as val, 'amylase' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE amylase IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bilirubin_total as val, 'bilirubin_total' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE bilirubin_total IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bilirubin_direct as val, 'bilirubin_direct' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE bilirubin_direct IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,bilirubin_indirect as val, 'bilirubin_indirect' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE bilirubin_indirect IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ck_cpk as val, 'ck_cpk' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE ck_cpk IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ggt as val, 'ggt' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE ggt IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ld_ldh as val, 'ld_ldh' as mark, hadm_id
    FROM mimiciv_derived.enzyme
    WHERE ld_ldh IS NOT NULL
    )
    )
    
    SELECT * FROM enzyme'''
    
    vitalsign_sql = '''WITH vitalsign AS ((
    SELECT t1.subject_id, charttime as time,storetime, heart_rate as val, 'heart_rate' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE heart_rate IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, sbp as val, 'sbp' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE sbp IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, dbp as val, 'dbp' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE dbp IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, mbp as val, 'mbp' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE mbp IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, resp_rate as val, 'resp_rate' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE resp_rate IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, temperature as val, 'temperature' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE temperature IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, spo2 as val, 'spo2' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE spo2 IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, glucose as val, 'glucose' as mark, hadm_id
    FROM mimiciv_derived.vitalsign t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE glucose IS NOT NULL
    )
    )
    SELECT * FROM vitalsign'''
    
    lab_rest_sql = '''WITH lab_rest AS ((
    SELECT subject_id,charttime as time,storetime,troponin_t as val, 'troponin_t' as mark, hadm_id
    FROM mimiciv_derived.cardiac_marker
    WHERE troponin_t IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ck_mb as val, 'ck_mb' as mark, hadm_id
    FROM mimiciv_derived.cardiac_marker
    WHERE ck_mb IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ntprobnp as val, 'ntprobnp' as mark, hadm_id
    FROM mimiciv_derived.cardiac_marker
    WHERE ntprobnp IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,d_dimer as val, 'd_dimer' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE d_dimer IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,fibrinogen as val, 'fibrinogen' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE fibrinogen IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,thrombin as val, 'thrombin' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE thrombin IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,inr as val, 'inr' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE inr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,pt as val, 'pt' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE pt IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,ptt as val, 'ptt' as mark, hadm_id
    FROM mimiciv_derived.coagulation
    WHERE ptt IS NOT NULL
    )
    UNION 
    (
    SELECT subject_id, starttime as time,starttime as storetime, weight as val, 'weight' as mark, hadm_id
    FROM mimiciv_derived.weight_durations t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE weight IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id,charttime as time,storetime,icp as val, 'icp' as mark, hadm_id
    FROM mimiciv_derived.icp t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE icp IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,crp as val, 'crp' as mark, hadm_id
    FROM mimiciv_derived.inflammation
    WHERE crp IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,valuenum as val, 'magnesium' as mark, hadm_id
    FROM mimiciv_hosp.labevents
    WHERE itemid = 50960 AND valuenum IS NOT NULL AND valuenum < 10
    )
    )
    
    
    SELECT * FROM lab_rest'''
    
    lab_demo_sql = '''WITH lab_demo AS ((
    SELECT t1.subject_id, gender, admission_age,height, hadm_id, admittime 
    FROM mimiciv_derived.icustay_detail t1 JOIN mimiciv_derived.height t2 
    ON t1.stay_id=t2.stay_id
    ))
    SELECT * FROM lab_demo'''
    
    lab_demo2_sql = '''WITH lab_demo2 AS ((
    SELECT subject_id, gender, admission_age,hadm_id, admittime 
    FROM mimiciv_derived.icustay_detail
    ))
    SELECT * FROM lab_demo2'''
    
    lab_demo3_sql = '''WITH lab_demo3 AS ((
    SELECT subject_id, gender, admission_age,hadm_id, first_icu_stay, icu_intime, icu_outtime, admittime, dischtime, dod 
    FROM mimiciv_derived.icustay_detail
    ))
    SELECT * FROM lab_demo3'''
    
    comorbidities_sql = '''WITH comorbidities AS ((
    SELECT * FROM mimiciv_derived.charlson 
    ))
    SELECT * FROM comorbidities'''
    
    cr_bl_sql = '''WITH cr_bl AS ((
    SELECT scr_min as val, 'scr_min' as mark, hadm_id
    FROM mimiciv_derived.creatinine_baseline
    WHERE scr_min IS NOT NULL
    )
    UNION
    (
    SELECT ckd as val, 'ckd' as mark, hadm_id
    FROM mimiciv_derived.creatinine_baseline
    WHERE ckd IS NOT NULL
    )
    UNION
    (
    SELECT mdrd_est as val, 'mdrd_est' as mark, hadm_id
    FROM mimiciv_derived.creatinine_baseline
    WHERE mdrd_est IS NOT NULL
    )
    UNION
    (
    SELECT scr_baseline as val, 'scr_baseline' as mark, hadm_id
    FROM mimiciv_derived.creatinine_baseline
    WHERE scr_baseline IS NOT NULL
    )
    )
    
    SELECT * FROM cr_bl'''
    
    gcs_sql = '''WITH gcs AS ((
    SELECT t1.subject_id, charttime as time,storetime, gcs as val, 'gcs' as mark, hadm_id
    FROM mimiciv_derived.gcs t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE gcs IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, gcs_motor as val, 'gcs_motor' as mark, hadm_id
    FROM mimiciv_derived.gcs t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE gcs_motor IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, gcs_verbal as val, 'gcs_verbal' as mark, hadm_id
    FROM mimiciv_derived.gcs t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE gcs_verbal IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, gcs_eyes as val, 'gcs_eyes' as mark, hadm_id
    FROM mimiciv_derived.gcs t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE gcs_eyes IS NOT NULL
    )
    UNION
    (
    SELECT t1.subject_id, charttime as time,storetime, gcs_unable as val, 'gcs_unable' as mark, hadm_id
    FROM mimiciv_derived.gcs t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE gcs_unable IS NOT NULL
    )
    )
    SELECT * FROM gcs'''
    
    kdigo_uo_sql = '''WITH kdigo_uo AS ((
    SELECT subject_id,charttime as time,urineoutput_6hr as val, 'urineoutput_6hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE urineoutput_6hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,urineoutput_12hr as val, 'urineoutput_12hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE urineoutput_12hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,urineoutput_6hr as val, 'urineoutput_24hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE urineoutput_24hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,uo_rt_6hr as val, 'uo_rt_6hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE uo_rt_6hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,uo_rt_12hr as val, 'uo_rt_12hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE uo_rt_12hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,uo_rt_24hr as val, 'uo_rt_24hr_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.kdigo_uo t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE uo_rt_24hr IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,urineoutput as val, 'urineoutput_kdigouo' as mark, hadm_id
    FROM mimiciv_derived.urine_output t1 JOIN mimiciv_icu.icustays t2
    ON t1.stay_id = t2.stay_id
    WHERE urineoutput IS NOT NULL
    )
    )
    
    SELECT * FROM kdigo_uo'''
    
    return bg_sql, blood_differential_sql, chemistry_sql, cbc_sql, enzyme_sql, vitalsign_sql, lab_rest_sql, lab_demo_sql, lab_demo2_sql, lab_demo3_sql, comorbidities_sql, cr_bl_sql, gcs_sql, kdigo_uo_sql

### FUNCTION TO LOAD DATA FROM MIMIC
def load_lab(table=None,Patients=None,cnx=None,sep_data=None):
    include_subject_id = tuple(sep_data["encounter_id"].to_list())
    curtime = time.time()
    print('loading {}'.format(table),end='...')
    data = pd.read_sql_query(table +' WHERE hadm_id IN {}'.format(include_subject_id), cnx).drop_duplicates().reset_index(drop=True)
    #data = pd.read_sql_query(table,cnx).drop_duplicates().reset_index(drop=True)
    print(f'({time.time()-curtime:.1f}s)')
    return data

def load_sepsis_dat(Patients=None,cnx=None, sep_data=None):
    #function loading all covariables
    bg_sql, blood_differential_sql, chemistry_sql, cbc_sql, enzyme_sql, vitalsign_sql, lab_rest_sql, lab_demo_sql, lab_demo2_sql, lab_demo3_sql, comorbidities_sql, cr_bl_sql, gcs_sql, kdigo_uo_sql = load_sepsis_dat_sql()
    
    lab_bg=load_lab(table=bg_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_bg=lab_bg.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    lab_bg.loc[[i == 'fio2_chartevents' for i in lab_bg['label']],'label']='fio2'
    lab_bg=lab_bg.drop_duplicates()
    
    lab_blood_differential=load_lab(table=blood_differential_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_blood_differential=lab_blood_differential.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_chemistry=load_lab(table=chemistry_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_chemistry=lab_chemistry.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_cbc=load_lab(table=cbc_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_cbc=lab_cbc.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_crbl=load_lab(table=cr_bl_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_crbl=lab_crbl.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_enzyme=load_lab(table=enzyme_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_enzyme=lab_enzyme.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_gcs=load_lab(table=gcs_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_gcs=lab_gcs.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_kdigo_uo=load_lab(table=kdigo_uo_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_kdigo_uo=lab_kdigo_uo.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_vitalsign=load_lab(table=vitalsign_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_vitalsign=lab_vitalsign.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_rest=load_lab(table=lab_rest_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_rest=lab_rest.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
#10 als Grenze für Magnesium
    
    lab_demo=load_lab(table=lab_demo_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_demo=lab_demo.rename(columns={"subject_id": "patient_id","time": "time_measured"})

    lab_demo2=load_lab(table=lab_demo2_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_demo2=lab_demo2.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    
    lab_demo3=load_lab(table=lab_demo3_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_demo3=lab_demo3.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})

    lab_comorbidities=load_lab(table=comorbidities_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    lab_comorbidities=lab_comorbidities.rename(columns={"subject_id": "patient_id"})
    
    lab_comp=pd.concat([lab_rest,lab_vitalsign, lab_kdigo_uo, lab_gcs, lab_enzyme, lab_cbc, lab_chemistry, lab_blood_differential,lab_bg])
    lab_comp=lab_comp.drop_duplicates()
    
    return lab_bg, lab_blood_differential,lab_chemistry, lab_cbc, lab_crbl, lab_enzyme, lab_gcs, lab_kdigo_uo, lab_vitalsign, lab_rest, lab_demo, lab_demo2, lab_demo3, lab_comorbidities, lab_comp
    
def load_CRT(Patients=None,cnx=None,sep_data=None):
    CRT_sql = '''WITH CRT AS ((
    SELECT subject_id,charttime as time,storetime,value as val, 'CRT r' as mark, hadm_id
    FROM mimiciv_icu.chartevents
    WHERE itemid = 223951 AND value IS NOT NULL
    )
    UNION
    (
    SELECT subject_id,charttime as time,storetime,value as val, 'CRT l' as mark, hadm_id
    FROM mimiciv_icu.chartevents
    WHERE itemid = 224308 AND value IS NOT NULL
    )
    )
    
    SELECT * FROM CRT'''
    
    CRT=load_lab(table=CRT_sql,Patients=Patients,cnx=cnx, sep_data=sep_data)
    CRT=CRT.rename(columns={"subject_id": "patient_id","time": "time_measured","val": "value","mark":"label"})
    CRT_r = CRT[CRT["label"]=="CRT r"]
    CRT_l = CRT[CRT["label"]=="CRT l"]
    d = {'Normal <3 Seconds': 0, 'Abnormal >3 Seconds': 1}
    CRT_l['value'] = CRT_l['value'].map(d)
    CRT_r['value'] = CRT_r['value'].map(d)
    CRT_ = CRT_r.copy()
    for i in range(CRT_r.shape[0]):
        a=CRT_l[(CRT_l["hadm_id"]==CRT_r["hadm_id"].iloc[i]) & (CRT_l["time_measured"] == CRT_r["time_measured"].iloc[i])]
        if len(a)>0 and a["value"].values[0]>CRT_r["value"].iloc[i]:
            CRT_["value"].iloc[i] = a["value"]
    for i in range(CRT_l.shape[0]):
        b=CRT_[(CRT_["hadm_id"]==CRT_l["hadm_id"].iloc[i]) & (CRT_["time_measured"] == CRT_l["time_measured"].iloc[i])]
        if len(b)==0:
            CRT_=CRT_.append(CRT_l.iloc[i])
    
    return CRT_

def abx_preproc(abx2):
    #Function for preprocessing and mapping Antibiotic Prescriptions
    abx2=abx2[abx2["starttime"]<=abx2["stoptime"]]
    abx2.loc[(abx2["antibiotic"].isin(["*NF* Ceftaroline"])),"antibiotic"]='Ceftaroline'
    abx2.loc[(abx2["antibiotic"].isin(["*NF* Moxifloxacin"])),"antibiotic"]='Moxifloxacin'
    abx2.loc[(abx2["antibiotic"].isin(["Amoxicillin-Clavulanate Susp."])),"antibiotic"]='Amoxicillin-Clavulanic Acid'
    abx2.loc[(abx2["antibiotic"].isin(["AMOXicillin Oral Susp."])),"antibiotic"]='Amoxicillin'
    abx2.loc[(abx2["antibiotic"].isin(["Amikacin Inhalation"])),"antibiotic"]='Amikacin'
    abx2.loc[(abx2["antibiotic"].isin(["Amphotericin B Nebulized"])),"antibiotic"]='Amphotericin B'
    abx2.loc[(abx2["antibiotic"].isin(["Ampicillin Sodium"])),"antibiotic"]='Ampicillin'
    abx2.loc[(abx2["antibiotic"].isin(["Azithromycin "])),"antibiotic"]='Azithromycin'
    abx2.loc[(abx2["antibiotic"].isin(["CeFAZolin"])),"antibiotic"]='Cefazolin'
    abx2.loc[(abx2["antibiotic"].isin(["CefazoLIN"])),"antibiotic"]='Cefazolin'
    abx2.loc[(abx2["antibiotic"].isin(["CeftriaXONE"])),"antibiotic"]='CefTRIAXone'
    abx2.loc[(abx2["antibiotic"].isin(["CeftazIDIME"])),"antibiotic"]='CefTAZidime'
    abx2.loc[(abx2["antibiotic"].isin(["Ceftolozane-Tazobactam *NF*"])),"antibiotic"]='Ceftolozane-Tazobactam'
    abx2.loc[(abx2["antibiotic"].isin(['Ceftazidime-Avibactam *NF*'])),"antibiotic"]='CefTAZidime-Avibactam (Avycaz)'
    abx2.loc[(abx2["antibiotic"].isin(['Ciprofloxacin IV'])),"antibiotic"]='Ciprofloxacin'
    abx2.loc[(abx2["antibiotic"].isin(['Ciprofloxacin HCl'])),"antibiotic"]='Ciprofloxacin'
    abx2.loc[(abx2["antibiotic"].isin(['Clindamycin Phosphate'])),"antibiotic"]='Clindamycin'
    abx2.loc[(abx2["antibiotic"].isin(['Clindamycin Solution'])),"antibiotic"]='Clindamycin'
    abx2.loc[(abx2["antibiotic"].isin(['Clindamycin Suspension'])),"antibiotic"]='Clindamycin'
    abx2.loc[(abx2["antibiotic"].isin(["Erythromycin Ethylsuccinate Suspension"])),"antibiotic"]='Erythromycin'
    abx2.loc[(abx2["antibiotic"].isin(["ERYTHROMYCIN"])),"antibiotic"]='Erythromycin'
    abx2.loc[(abx2["antibiotic"].isin(["Gentamicin 2.5 mg/mL in Sodium Citrate 4%"])),"antibiotic"]='Gentamicin'
    abx2.loc[(abx2["antibiotic"].isin(["Gentamicin Intraventricular"])),"antibiotic"]='Gentamicin'
    abx2.loc[(abx2["antibiotic"].isin(["Gentamicin Sulfate"])),"antibiotic"]='Gentamicin'
    abx2.loc[(abx2["antibiotic"].isin(["LevoFLOXacin"])),"antibiotic"]='Levofloxacin'
    abx2.loc[(abx2["antibiotic"].isin(["Linezolid Suspension"])),"antibiotic"]='Linezolid'
    abx2.loc[(abx2["antibiotic"].isin(["MetRONIDAZOLE"])),"antibiotic"]='MetroNIDAZOLE'
    abx2.loc[(abx2["antibiotic"].isin(["MetRONIDAZOLE (FLagyl)"])),"antibiotic"]='MetroNIDAZOLE'
    abx2.loc[(abx2["antibiotic"].isin(["Mupirocin "])),"antibiotic"]='Mupirocin'
    abx2.loc[(abx2["antibiotic"].isin(["Mupirocin Ointment 2%"])),"antibiotic"]='Mupirocin Nasal Ointment 2%'
    abx2.loc[(abx2["antibiotic"].isin(['Nitrofurantoin Monohyd (MacroBID)'])),"antibiotic"]='Nitrofurantoin'
    abx2.loc[(abx2["antibiotic"].isin(['Nitrofurantoin (Macrodantin)'])),"antibiotic"]='Nitrofurantoin'
    abx2.loc[(abx2["antibiotic"].isin(['Penicillin G Benzathine'])),"antibiotic"]='Penicillin'
    abx2.loc[(abx2["antibiotic"].isin(['Penicillin G Potassium'])),"antibiotic"]='Penicillin'
    abx2.loc[(abx2["antibiotic"].isin(['Penicillin V Potassium'])),"antibiotic"]='Penicillin'
    abx2.loc[(abx2["antibiotic"].isin(['Piperacillin-Tazobactam Na'])),"antibiotic"]='Piperacillin-Tazobactam'
    abx2.loc[(abx2["antibiotic"].isin(['RifAMPin'])),"antibiotic"]='Rifampin'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfameth/Trimethoprim '])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfameth/Trimethoprim SS'])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfameth/Trimethoprim DS'])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfameth/Trimethoprim SS'])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfameth/Trimethoprim Suspension'])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Sulfamethoxazole-Trimethoprim'])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['Tetracycline HCl'])),"antibiotic"]='Tetracycline'
    abx2.loc[(abx2["antibiotic"].isin(['Tobramycin Inhalation Soln'])),"antibiotic"]='Tobramycin'
    abx2.loc[(abx2["antibiotic"].isin(['Tobramycin Sulfate'])),"antibiotic"]='Tobramycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin '])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin Enema'])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin Intrathecal'])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin Intraventricular'])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin Ora'])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Vancomycin Oral Liquid'])),"antibiotic"]='Vancomycin'
    abx2.loc[(abx2["antibiotic"].isin(['Zithromax Z-Pak'])),"antibiotic"]='Azithromycin'
    abx2.loc[(abx2["antibiotic"].isin(['azithromycin'])),"antibiotic"]='Azithromycin'
    abx2.loc[(abx2["antibiotic"].isin(["ceFAZolin"])),"antibiotic"]='Cefazolin'
    abx2.loc[(abx2["antibiotic"].isin(["ciprofloxacin"])),"antibiotic"]='Ciprofloxacin'
    abx2.loc[(abx2["antibiotic"].isin(["moxifloxacin"])),"antibiotic"]='Moxifloxacin'
    abx2.loc[(abx2["antibiotic"].isin(["nitrofurantoin macrocrystal"])),"antibiotic"]='Nitrofurantoin'
    abx2.loc[(abx2["antibiotic"].isin(["sulfamethoxazole-trimethoprim"])),"antibiotic"]='Sulfameth/Trimethoprim'
    abx2.loc[(abx2["antibiotic"].isin(['vancomycin'])),"antibiotic"]='Vancomycin'
    return abx2

def patient_stay_search_antibiotic(antibiotic_string=None,antibiotic_id=None,prescriptions=None,icu_inputevents=None,icustays=None):
    # Returns all hadm ids, stay numbers and stay times of the patients only treated with the specified antibiotics
    # antibiotic_id: List of IDs of the included antibiotics
    # antibiotic_string: String containing the antibiotics, where "|" separates the antibiotics
    # icustays: Tensor including detailed information on ICU stay as provided by OpenSep
    
    antibiotic_hadms_prescriptions=prescriptions[(prescriptions["antibiotic"].str.contains(antibiotic_string)) & (~prescriptions["antibiotic"].str.contains("Lock|Graded"))]
    icustays["icu_intime"]=icustays["icu_intime"].dt.tz_localize(None)
    icustays["icu_outtime"]=icustays["icu_outtime"].dt.tz_localize(None)
    
    hadm_list=[]
    stay_list=[]
    stay_times=[]
    for i in range(len(antibiotic_hadms_prescriptions)):
        antibiotic_index = antibiotic_hadms_prescriptions.iloc[i]
        help_hadm = antibiotic_index["hadm_id"]
        prescriptions_hadm_no = prescriptions[(prescriptions["hadm_id"]==help_hadm) & (~prescriptions["antibiotic"].str.contains(antibiotic_string))].copy()
        icu_inputevents_hadm_yes = icu_inputevents[(icu_inputevents["hadm_id"]==help_hadm) & (icu_inputevents["itemid"].isin(antibiotic_id))]
        icu_inputevents_hadm_no = icu_inputevents[(icu_inputevents["hadm_id"]==help_hadm) & (~icu_inputevents["itemid"].isin(antibiotic_id))  & (icu_inputevents["ordercomponenttypedescription"] == "Main order parameter")]
        adt_hadm_icu = icustays[icustays["hadm_id"].astype(int)==help_hadm].sort_values("icu_intime")   
        
        #create list for patient stays... 
        stays_list_patient=[]
        stays_list_help=[]
        stays_ids_patient=[]
        for j in range(len(adt_hadm_icu)):
            if j < len(adt_hadm_icu)-1:
                if (adt_hadm_icu.iloc[j+1]["icu_intime"]-adt_hadm_icu.iloc[j]["icu_outtime"])<np.timedelta64(24,'h'):
                    if j in stays_list_help:
                        del stays_list_help[-1]
                        stays_list_patient.append([stays_list_patient[-1][0],j+1])
                        stays_list_help.append(j+1)
                        del stays_list_patient[-2]
                        stays_ids_patient[-1].append(adt_hadm_icu.iloc[j+1]["stay_id"])
                    else:
                        stays_list_patient.append([j,j+1])
                        stays_list_help.append(j)
                        stays_list_help.append(j+1)
                        stays_ids_patient.append([adt_hadm_icu.iloc[j]["stay_id"],adt_hadm_icu.iloc[j+1]["stay_id"]])
                elif (len(stays_list_patient)>0 and len(stays_list_patient[-1])==2 and stays_list_patient[-1][1]!=j) or (len(stays_list_patient)>0 and len(stays_list_patient[-1])==1 and stays_list_patient[-1]!=j) or len(stays_list_patient)==0:
                    stays_list_patient.append([j])
                    stays_list_help.append(j)
                    stays_ids_patient.append([adt_hadm_icu.iloc[j]["stay_id"]])
            elif (len(stays_list_patient)>0 and len(stays_list_patient[-1])==2 and stays_list_patient[-1][1]!=j) or (len(stays_list_patient)>0 and len(stays_list_patient[-1])==1 and stays_list_patient[-1]!=j) or (len(stays_list_patient)==0):
                    stays_list_patient.append([j])
                    stays_list_help.append(j)
                    stays_ids_patient.append([adt_hadm_icu.iloc[j]["stay_id"]])

        for j in range(len(stays_list_patient)):
            if len(stays_list_patient[j])>1:
                adt_hadm_j_start = np.datetime64(adt_hadm_icu.iloc[stays_list_patient[j][0]]["icu_intime"])
                adt_hadm_j_end = np.datetime64(adt_hadm_icu.iloc[stays_list_patient[j][1]]["icu_outtime"])
            else:
                adt_hadm_j_start = adt_hadm_icu.iloc[stays_list_patient[j]]["icu_intime"].values[0]
                adt_hadm_j_end = adt_hadm_icu.iloc[stays_list_patient[j]]["icu_outtime"].values[0]
            stay_ids = stays_ids_patient[j]
            icu_inputevents_stay_no=icu_inputevents_hadm_no[icu_inputevents_hadm_no["stay_id"].isin(stay_ids)]
            icu_inputevents_stay_yes=icu_inputevents_hadm_yes[icu_inputevents_hadm_yes["stay_id"].isin(stay_ids)]
            

            #Check whether further Antibiotic was administrated at ICU
            if not ((((prescriptions_hadm_no["starttime"]>(adt_hadm_j_start-np.timedelta64(12,'h'))) & (prescriptions_hadm_no["starttime"]<adt_hadm_j_end) & (prescriptions_hadm_no["stoptime"]>(adt_hadm_j_start + np.timedelta64(6,'h')))).any()) or ((prescriptions_hadm_no["stoptime"]>(adt_hadm_j_start+np.timedelta64(12,'h'))) & (prescriptions_hadm_no["starttime"]<(adt_hadm_j_end-np.timedelta64(12,'h')))).any()):
                    #Check if diff. medication was given during icu inputevents
                if len(icu_inputevents_stay_no)==0:
                    #check, if starttime or stoptime during icu stay
                    #12 hours before and 24 hours after
                    if ((antibiotic_index["starttime"]>(adt_hadm_j_start-np.timedelta64(12,'h'))) & (antibiotic_index["starttime"]<adt_hadm_j_end) & (antibiotic_index["stoptime"]>(adt_hadm_j_start + np.timedelta64(6,'h')))) or ((antibiotic_index["stoptime"]>(adt_hadm_j_start+np.timedelta64(12,'h'))) & (antibiotic_index["starttime"]<(adt_hadm_j_end-np.timedelta64(12,'h')))):
                        if not stay_ids in stay_list:
                            hadm_list.append(help_hadm)
                            stay_list.append(stay_ids)
                            stay_times.append([adt_hadm_j_start,adt_hadm_j_end])
                    #check if administration by inputevents
                    elif (len(icu_inputevents_stay_yes)>0):
                        if not stay_ids in stay_list:
                            hadm_list.append(help_hadm)
                            stay_list.append(stay_ids)
                            stay_times.append([adt_hadm_j_start,adt_hadm_j_end])
    return hadm_list, stay_list, stay_times

def get_antibiotic_administrations(antibiotic_stay_list,antibiotic_id,antibiotic_string,prescriptions,icu_inputevents):
    # antibiotic_stay_list: List of ICU stays
    # antibiotic_id: List of IDs of the included antibiotics
    # antibiotic_string: String containing the antibiotics, where "|" separates the antibiotics
    # Returns tensor of antibiotic administration for the patient
    
    antibiotic_abx_list=[]
    for i in antibiotic_stay_list:
        i_abx=icu_inputevents[(icu_inputevents["stay_id"].isin(i)) & (icu_inputevents["itemid"].isin(antibiotic_id))]
        if len(i_abx)>0:
            hadm = i_abx["hadm_id"].iloc[0]
            for k in range(i_abx.shape[0]):
                add=True
                j=i_abx.iloc[k]
                time = j["starttime"]
                dat_prescriptions = prescriptions[(prescriptions["hadm_id"]==hadm) & (prescriptions["antibiotic"].str.contains(antibiotic_string)) & (~prescriptions["antibiotic"].str.contains("Lock|Graded")) & (prescriptions["starttime"]<=time) & (prescriptions["stoptime"]>=time)]
                if j["amountuom"]=="dose" and len(dat_prescriptions) >0:
                    j["amount"] = j["amount"]*float(dat_prescriptions["dose_val_rx"].iloc[0])
                    j["amountuom"] = dat_prescriptions["dose_unit_rx"].iloc[0]
                elif j["amountuom"]=="dose":
                    dat_prescriptions = prescriptions[(prescriptions["hadm_id"]==hadm) & (prescriptions["antibiotic"].str.contains(antibiotic_string)) & (~prescriptions["antibiotic"].str.contains("Lock|Graded")) & (prescriptions["starttime"]<=(time+np.timedelta64(6,'h'))) & (prescriptions["stoptime"]>=(time-np.timedelta64(6,'h')))]
                    if len(dat_prescriptions)>0:
                        #print(1)
                        j["amount"] = j["amount"]*float(dat_prescriptions["dose_val_rx"].iloc[0])
                        j["amountuom"] = dat_prescriptions["dose_unit_rx"].iloc[0]
                    else:
                        dat_prescriptions = prescriptions[(prescriptions["hadm_id"]==hadm) & (prescriptions["antibiotic"].str.contains(antibiotic_string)) & (~prescriptions["antibiotic"].str.contains("Lock|Graded")) & (prescriptions["starttime"]<=(time+np.timedelta64(24,'h'))) & (prescriptions["stoptime"]>=(time-np.timedelta64(24,'h')))]
                        if len(dat_prescriptions)>0:
                            #print(2)
                            j["amount"] = j["amount"]*float(dat_prescriptions["dose_val_rx"].iloc[0])
                            j["amountuom"] = dat_prescriptions["dose_unit_rx"].iloc[0]
                        else:
                            #print(i)
                            #print(3)
                            add=False
                if add and j["amount"]!=0:
                    if j["amountuom"]=='gm':
                        j["amountuom"]='g'
                    elif j["amountuom"]=='grams':
                        j["amountuom"]='g'
                    elif j["amountuom"]=='mg':
                        j["amountuom"]='g'
                        j["amount"]=j["amount"]/1000
                    j["amount"]=np.round(j["amount"],decimals=3)
                    antibiotic_abx_list.append(j)
    antibiotic_abx_df=pd.DataFrame(antibiotic_abx_list)
    
    return antibiotic_abx_df

def round_time_func(lab_hadm, round_minutes):
    #removing changed timestamps from options of timestamps
    changed_list=[]
    lab_hadm2=lab_hadm.copy()
    for i in range(len(lab_hadm2["time_measured"].unique())):
        time_a = lab_hadm2["time_measured"].unique()[i]
        if time_a not in changed_list:
            for j in range(i+1,len(lab_hadm["time_measured"].unique())):
                time_b = lab_hadm["time_measured"].unique()[j]
                if time_b<time_a+np.timedelta64(round_minutes,'m'):
                    #print(time_a)
                    #print(time_b)
                    lab_hadm2.loc[lab_hadm2["time_measured"]==time_b,'time_measured']=time_a
                    changed_list.append(time_b)
    lab_hadm=lab_hadm2.copy()
    return lab_hadm

def aggregate_startvalues_func(lab_hadm, start_hours):
    lab_hadm2=lab_hadm.copy()
    for i in range(len(lab_hadm["time_measured"].unique())):
        time_a = lab_hadm["time_measured"].unique()[i]
        if start_hours is None:
            if time_a <np.timedelta64(0,'h'):
                lab_hadm2.loc[(lab_hadm2["time_measured"]==time_a) & (lab_hadm2["label"]!="SOFA"),'time_measured']=np.timedelta64(0,'h')
        else:
            #print(time_a)
            #print(np.timedelta(0,'h'))
            if time_a<np.timedelta64(0,'h') and time_a>=np.timedelta64(-(int(start_hours*60)),'m'):
                lab_hadm2.loc[(lab_hadm2["time_measured"]==time_a) & (lab_hadm2["label"]!="SOFA"),'time_measured']=np.timedelta64(0,'h')
    lab_hadm=lab_hadm2.copy()
    return lab_hadm

def to_hours(a):
    return a.total_seconds()/3600

def round_nearest_func(x, a):
    return round(x / a) * a

def fill_forward(x, max_length):
    return torch.cat([x, x[-1].unsqueeze(0).expand(max_length - x.size(0), x.size(1))])

def one_patient_to_tensor(lab_df, index=0, hadm_id=None, thresh=None, round_time=False, round_nearest=True, round_minutes=15, aggregate_startvalues=False, remove_unaggregated_values=False, max_time=None, start_hours=None, timetype="time_measured", start_timepoint='sepsis', missing_mask=True, sep_data=None, lab_demo=None, variables=None, just_icu=False, icustays=None,stay=None, antibiotics=None, remove_noant=False,antibiotics_variables=None,binary_antibiotics=False,static_bin_ant=False):
    # description of the variables see multiple_patients_predictions_tensor
    
    lab_df2=lab_df.groupby("label")["hadm_id"].nunique()
    #selecting variables
    if thresh is not None:
        variables_list=lab_df2[lab_df2>(len(lab_df["hadm_id"].unique()))*thresh].keys()
    if variables is not None:
        variables_list=variables
    
    if index is not None and hadm_id is None:
    #selecting hadm_i
        hadm_id = np.sort(sep_data["encounter_id"].unique().astype(int))[index]

    if stay is not None:
        hadm_id = icustays[icustays["stay_id"]==stay[0]]["hadm_id"].iloc[0]    
    
    #all laboratory values of one hadm_id
    lab_hadm = lab_df[lab_df["hadm_id"]==hadm_id]
    #Just three columns needed
    lab_hadm = lab_hadm[[timetype,"value","label"]]
    lab_hadm["label"]=lab_hadm["label"].astype("string")
    lab_hadm=lab_hadm[lab_hadm["label"].isin(variables_list)]    
    
    lab_hadm = lab_hadm.drop_duplicates()
    #Sorting of labvalues
    lab_hadm=lab_hadm.sort_values(by=timetype)
    
    if just_icu:
        if stay is not None:
            icu_hadm=icustays[icustays["stay_id"].isin(stay)]
        else:
            icu_hadm=icustays[icustays["hadm_id"]==hadm_id]
        lab_hadm=lab_hadm[(lab_hadm["time_measured"]>(min(icu_hadm["icu_intime"])-np.timedelta64(int(start_hours*60),'m'))) & (lab_hadm["time_measured"]<max(icu_hadm["icu_outtime"]))]# & (lab_hadm["time_measured"]<max(icu_hadm["dischtime"]))]
    
    if start_timepoint=='sepsis':
        start = pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0]
    elif start_timepoint=='sepsis_icu':
        start = max(pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0],min(lab_hadm[lab_hadm["label"]=='SOFA']["time_measured"]))
    elif start_timepoint=="icu":
        if stay is None:
            start = min(lab_demo["icu_intime"][lab_demo["hadm_id"]==hadm_id])
        else:
            start = min(icu_hadm["icu_intime"])
    elif start_timepoint=='admission':
        start = min(lab_demo["admittime"][lab_demo["hadm_id"]==hadm_id])
    lab_hadm[timetype]=lab_hadm[timetype]-start
    
    if round_time:
        lab_hadm = round_time_func(lab_hadm, round_minutes)
    
    #aggregation of startvalues
    if aggregate_startvalues:
        lab_hadm = aggregate_startvalues_func(lab_hadm, start_hours)
    if remove_unaggregated_values:
        lab_hadm=lab_hadm[(lab_hadm["time_measured"] >= np.timedelta64(0,'h'))]
        if max_time is not None:
            lab_hadm=lab_hadm[(lab_hadm["time_measured"] <= np.timedelta64(max_time,'h'))]
        
    lab_hadm["time_measured"] = lab_hadm["time_measured"].map(to_hours)
    if round_nearest:
        lab_hadm["time_measured"] = lab_hadm["time_measured"].apply(lambda x: round_nearest_func(x, a=round_minutes/60))
    
    lab_hadm_old=lab_hadm.copy()
    lab_hadm=lab_hadm.pivot_table(index=timetype,columns="label",values="value")
    if lab_hadm.empty:
        return None, None, None
    else:
        lab_hadm=lab_hadm.reindex(index=range(int(min(lab_hadm.index)),int(max(lab_hadm.index))+1))
    if "CRT" in variables_list and "CRT" in lab_hadm_old.keys():
        CRT = lab_hadm_old.pivot_table(index=timetype,columns="label",values="value",aggfunc=max)["CRT"]
        CRT=CRT.reindex(index=range(int(min(lab_hadm.index)),int(max(lab_hadm.index))+1))
        lab_hadm["CRT"]=CRT
    if "SOFA" in variables_list:
        def nearest(items, pivot):
            return min(items, key=lambda x: abs(x - pivot))
        lab_SOFA=[]
        lab_hadm_SOFA=lab_hadm_old[lab_hadm_old["label"]=="SOFA"]
        for i in lab_hadm.index:
            time=nearest(lab_hadm_SOFA["time_measured"],i)
            lab_SOFA.append(lab_hadm_SOFA[lab_hadm_SOFA["time_measured"]==time]["value"].iloc[0])
        lab_hadm["SOFA"]=lab_SOFA
    
    for i in variables_list:
        if i not in lab_hadm.keys():
            lab_hadm[i] = np.nan
    lab_hadm = lab_hadm[variables_list]
    
    #Preprocessing of Antibiotic data
    if antibiotics is not None:
        
        if binary_antibiotics:
            pres_hadm = antibiotics[antibiotics["hadm_id"]==hadm_id]
            antibiotics_array = np.zeros(shape=[lab_hadm.shape[0],len(antibiotics_variables)])
            
            if static_bin_ant:
                static_ant = np.zeros(shape=[len(antibiotics_variables)])
                
                pres_hadm_bin = pres_hadm.copy()
                pres_hadm_bin = pres_hadm_bin.sort_values("stoptime",ascending=False)
                for i in range(len(antibiotics_variables)):
                    pres_hadm_bin_ant = pres_hadm_bin[pres_hadm_bin["label"]==antibiotics_variables[i]]
                    helpval=np.nan
                    for j in range(pres_hadm_bin_ant.shape[0]):
                        a=pres_hadm_bin_ant.iloc[j]["starttime"]-start
                        a=int(np.round(to_hours(a)))
                        b=pres_hadm_bin_ant.iloc[j]["stoptime"]-start
                        b=int(np.round(to_hours(b)))
                        if a < 0 and b>-12:
                            helpval=a
                        if not np.isnan(helpval) and b>helpval-12 and a < helpval:
                            helpval=a
                    if not np.isnan(helpval):
                        static_ant[i] = helpval

            else:
                static_ant=None
            
                
            for i in range(pres_hadm.shape[0]):
                index = antibiotics_variables.index(pres_hadm.iloc[i]["label"])
                a=pres_hadm.iloc[i]["starttime"]-start
                a=int(np.round(to_hours(a)))
                if a<0:
                    a=0
                b=pres_hadm.iloc[i]["stoptime"]-start
                b=int(np.round(to_hours(b)))
                if b>0:
                    antibiotics_array[a:b,index]=1
                
                #if there is a pause of <12h in between two prescriptions, then setting prescriptions to 1
                for j in range(antibiotics_array.shape[1]):
                    q=np.where(antibiotics_array[:,j]==1)
                    dq = np.diff(q[0])-1
                    for k in range(dq.shape[0]):
                        if dq[k]>0 and dq[k]<12:
                            antibiotics_array[q[0][k]:q[0][k+1],j]=1
                
            if remove_noant and (antibiotics_array==0).all():
                return None,None,None
            
            variables_list=variables_list+antibiotics_variables
            
            antibiotics_pd=pd.DataFrame(antibiotics_array,columns=antibiotics_variables)
            lab_hadm = pd.concat([lab_hadm,antibiotics_pd],axis=1)
                
        else:
            antibiotics_stay = antibiotics[antibiotics["stay_id"].isin(stay)].sort_values(by=timetype)
            antibiotics_stay = antibiotics_stay[[timetype,"value","label"]]
            antibiotics_stay["time_measured"]=antibiotics_stay["time_measured"]-start
            
            #aggregation of antibiotic data for the last 12 hours before start
            if aggregate_startvalues:
                antibiotics_stay = aggregate_startvalues_func(antibiotics_stay, 12)
                if remove_unaggregated_values:
                    antibiotics_stay=antibiotics_stay[antibiotics_stay["time_measured"] >= np.timedelta64(0,'h')]
                    if max_time is not None:
                        antibiotics_stay=antibiotics_stay[antibiotics_stay["time_measured"] <= np.timedelta64(max_time,'h')]
                
                antibiotics_stay["time_measured"] = antibiotics_stay["time_measured"].map(to_hours)
                if round_nearest:
                    antibiotics_stay["time_measured"] = antibiotics_stay["time_measured"].apply(lambda x: round_nearest_func(x, a=round_minutes/60))
        
            antibiotics_stay=antibiotics_stay.pivot_table(index=timetype,columns="label",values="value",aggfunc=sum)
            if antibiotics_variables is not None:
                for i in antibiotics_variables:
                    if i not in antibiotics_stay.keys():
                        antibiotics_stay[i]=np.nan
                antibiotics_stay=antibiotics_stay[antibiotics_variables]
            
            if not antibiotics_stay.empty:
                lab_hadm = lab_hadm.merge(antibiotics_stay,how='outer',left_index=True,right_index=True)
                lab_hadm[lab_hadm[antibiotics_stay.keys()].isna()]=0.0
            else:
                if remove_noant:
                    return None, None, None
                else:
                    lab_hadm[antibiotics["label"]]=0
            variables_list=variables_list+list(antibiotics_stay.keys())
            static_ant=None
                
    if missing_mask:    
        mask_df = lab_hadm[variables].notna().cumsum(axis=0).astype(float)
        a=[s + "_mask" for s in list(mask_df.keys())]
        mask_df=mask_df.set_axis(a,axis=1)
        lab_hadm = pd.concat([lab_hadm,mask_df],axis=1)
        
    lab_hadm = lab_hadm.reset_index()
    return lab_hadm, variables_list, static_ant

def one_patient_mean_per_hour(lab_df, index=None, hadm_id=None, round_nearest=True, round_minutes=60, aggregate_startvalues=True, remove_unaggregated_values=False, max_time=None, start_hours=0.5, start_timepoint='sepsis', sep_data=None, lab_demo=None, variables=None, just_icu=False, icustays=None, stay=None):
    #lab_df is a pd dataframe consisting of patient_id
    #time_measured in datetimeformat
    #value in float
    #label of lab values in string
    
    #hadm_id as hospital admission id
    #index is the index of hadm_id
    #stay is the stay id
    #one of those mearues should be provided
    #other variables as in multiple patients mean
   
    if index is not None and hadm_id is None:
        #selecting hadm_id is it is None
        hadm_id = np.sort(sep_data["encounter_id"].unique().astype(int))[index]
    if stay is not None:
        hadm_id = icustays[icustays["stay_id"]==stay[0]]["hadm_id"].iloc[0]
    
    #all laboratory values of one hadm_id
    lab_hadm = lab_df[lab_df["hadm_id"]==hadm_id]
    lab_hadm=lab_hadm[lab_hadm["label"].isin(variables)]    
    lab_hadm["label"]=lab_hadm["label"].astype("string")
    #Just three columns needed
    lab_hadm = lab_hadm[["time_measured","value","label"]]
    lab_hadm = lab_hadm.drop_duplicates()
    #Sorting of labvalues
    lab_hadm=lab_hadm.sort_values(by="time_measured")
    
    if just_icu:
        if stay is not None:
            icu_hadm=icustays[icustays["stay_id"].isin(stay)]
        else:
            icu_hadm=icustays[icustays["hadm_id"]==hadm_id]
        if len(icu_hadm)>0:
            lab_hadm=lab_hadm[(lab_hadm["time_measured"]>(min(icu_hadm["icu_intime"])-np.timedelta64(int(start_hours*60),'m'))) & (lab_hadm["time_measured"]<max(icu_hadm["icu_outtime"]))]
        else:
            return None, None
            
    if start_timepoint=='sepsis':
    #Sepsis_time starting time 
        start = pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0]
    elif start_timepoint=='sepsis_icu':
        start = max(pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0],min(lab_hadm[lab_hadm["label"]=='SOFA']["time_measured"]))
    elif start_timepoint=='icu':
        start = min(lab_demo["icu_intime"][lab_demo["hadm_id"]==hadm_id])
    elif start_timepoint=='admission':
        start = min(lab_demo["admittime"][lab_demo["hadm_id"]==hadm_id])
    lab_hadm["time_measured"]=lab_hadm["time_measured"]-start
    
    if aggregate_startvalues:
        lab_hadm = aggregate_startvalues_func(lab_hadm, start_hours)
    if remove_unaggregated_values:
        lab_hadm=lab_hadm[lab_hadm["time_measured"] >= np.timedelta64(0,'h')]
        if max_time is not None:
            lab_hadm=lab_hadm[lab_hadm["time_measured"] <= np.timedelta64(max_time,'h')]
        
    lab_hadm["time_measured"] = lab_hadm["time_measured"].map(to_hours)
    if round_nearest:
        lab_hadm["time_measured"] = lab_hadm["time_measured"].apply(lambda x: round_nearest_func(x, a=round_minutes/60))
    
    lab_hadm_old=lab_hadm.copy()
    lab_hadm=lab_hadm.pivot_table(index="time_measured",columns="label",values="value")
    if "CRT" in variables and "CRT" in lab_hadm_old.keys():
        CRT = lab_hadm_old.pivot_table(index="time_measured",columns="label",values="value",aggfunc=max)["CRT"]
        lab_hadm["CRT"]=CRT
    
    for i in variables:
        if i not in lab_hadm.keys():
            lab_hadm[i] = np.nan
    #sort the columns of the data
    lab_hadm = lab_hadm[variables]

    lab_hadm = lab_hadm.reset_index()
    
    return lab_hadm, variables

def fill_missing_range(df, field, range_from, range_to, range_step=1, fill_with=0):
    return df\
      .merge(how='right', on=field,
            right = pd.DataFrame({field:np.arange(range_from, range_to, range_step)}))\
      .sort_values(by=field).reset_index().fillna(fill_with).drop(['index'], axis=1)

def multiple_patients_mean(lab_df, hadm_ids=None, round_nearest=True, round_minutes=60, aggregate_startvalues=True, remove_unaggregated_values=False, max_time=None, start_hours=0.5, start_timepoint='sepsis', sep_data=None, lab_demo=None, variables=None, thresh=0.5, just_icu=False, icustays=None, stays_list=None):
    # lab_df: DataFrame containing all covariables
    # hadm_ids: Subset of all patients' Hospital Admission IDs
    # round_nearest: True if time should be rounded
    # round_minutes: Rounding to the nearest x minutes
    # aggregate_startvalues: True if values measured y before the start should be considered or not
    # remove_unaggregated_values: Whether to remove unaggregated values
    # max_time: Maximum time in hours when computation is stopped
    # start_hours: Time duration for considering values before the starting timepoint, if aggregation is done
    # start_timepoint: Different options for selecting data start time - "sepsis_icu": Latest of first measured SOFA-Score at ICU and Sepsis onset, 'icu': First measured SOFA-Score, 'sepsis': Sepsis onset, 'admission': All data
    # sep_data: DataFrame including SOFA-Scores and Sepsis onset
    # lab_demo: DataFrame including static variables
    # just_icu: True if only data measured at ICU is considered
    # icustays: Tensor including detailed information on ICU stay as provided by OpenSep

    tensor_list=[]
    
    lab_df2=lab_df.groupby("label")["hadm_id"].nunique()
    if variables is None:
        variables=lab_df2[lab_df2>(len(lab_df["hadm_id"].unique()))*thresh].keys()
    
    if stays_list is None and hadm_ids is not None:    
        for i in hadm_ids:    
            lab_hadm_patient,variables_list=one_patient_mean_per_hour(lab_df=lab_df, index=None, hadm_id=i, round_nearest=round_nearest, round_minutes=round_minutes, aggregate_startvalues=aggregate_startvalues, remove_unaggregated_values=remove_unaggregated_values, max_time=max_time, start_hours=start_hours, start_timepoint=start_timepoint, sep_data=sep_data, lab_demo=lab_demo, variables=variables,just_icu=just_icu, icustays=icustays)
            if lab_hadm_patient is not None:
                lab_hadm_patient=fill_missing_range(lab_hadm_patient, 'time_measured', 0.0, max_time+1, round_minutes/60, np.nan)
                tensor_list.append(torch.from_numpy(lab_hadm_patient.to_numpy().astype(np.float32)))
    elif stays_list is not None and hadm_ids is None:
        for i in stays_list:    
            lab_hadm_patient,variables_list=one_patient_mean_per_hour(lab_df=lab_df, index=None, hadm_id=None, round_nearest=round_nearest, round_minutes=round_minutes, aggregate_startvalues=aggregate_startvalues, remove_unaggregated_values=remove_unaggregated_values, max_time=max_time, start_hours=start_hours, start_timepoint=start_timepoint, sep_data=sep_data, lab_demo=lab_demo, variables=variables,just_icu=just_icu, icustays=icustays, stay=i)
            lab_hadm_patient=fill_missing_range(lab_hadm_patient, 'time_measured', 0.0, max_time+1, round_minutes/60, np.nan)
            tensor_list.append(torch.from_numpy(lab_hadm_patient.to_numpy().astype(np.float32)))
        
    lab = torch.stack(tensor_list)
    
    return lab, variables_list



def multiple_patients_predictions_tensor(lab_df, min_pred=0,max_pred=None, pred_times_in_h=1, thresh=0.5, round_time=False, round_nearest=True, round_minutes=15, aggregate_startvalues=False, start_hours=None, remove_unaggregated_values=False, timetype="time_measured", start_timepoint='sepsis', missing_mask=True, sep_data=None, lab_demo=None, list_of_hadms=None,first_adms=None,variables=None,print_=False,just_icu=False, icustays=None, stays_list=None, missing_imputation_start=True,antibiotics=None, remove_noant=False, static=None, static_time=False, standardize=False, train_test_split=False, seed=None,antibiotics_variables=None,binary_antibiotics=False,static_bin_ant=False):
    # min_pred: Start timepoint for predictions, default: 0 (offset is mean)
    # max_pred: Endpoint for predictions (reducing memory usage), default: None
    # pred_times_in_h: Intervals where predictions are performed in hours (due to exploding memory usage due to interpolation scheme of Neural CDEs), default: 1h
    # lab_df: DataFrame containing all covariables
    # thresh: Threshold describing the maximum allowed percentage of missing values; otherwise, variables will be removed. Default: None
    # sep_data: DataFrame including SOFA-Scores and Sepsis onset
    # round_time: Rounding to real hours (not needed for the Mimic data), default: False
    # round_nearest: True if time should be rounded
    # round_minutes: Rounding to the nearest x minutes
    # aggregate_startvalues: True if values measured y before the start should be considered or not
    # start_hours: Time duration for considering values before the starting timepoint, if aggregation is done
    # remove_unaggregated_values: Whether to remove unaggregated values
    # start_timepoint: Different options for selecting data start time - "sepsis_icu": Latest of first measured SOFA-Score at ICU and Sepsis onset, 'icu': First measured SOFA-Score, 'sepsis': Sepsis onset, 'admission': All data
    # lab_demo: DataFrame including static variables
    # print_: Printing patient ids
    # list_of_hadms: List of all patients' Hospital Admission IDs (alternatively stays_list can be specified)
    # just_icu: True if only data measured at ICU is considered
    # icustays: Tensor including detailed information on ICU stay as provided by OpenSep
    # stays_list: List of considered stay ids (alternatively list_of_hadms can be specified)
    # variables: List containing all covariables (and outcomes)
    # missing_imputation_start: NeuralCDEs needing a missing value imputation at the start, default: True
    # antibiotics: Antibiotic prescriptions data as computed by the OpenSep Pipeline
    # remove_noant: Remove patients to whom no Antibiotics are administered
    # static: DataFrame containing static data with the (default) column names ["admission_age", "height", "weight", "male"]
    # static_time: Include static variables as (time-dependent) DYNAMIC variables, default: False
    # standardize: Describes, whether a standardization is performed or not, default: True
    # train_test_split: Describes, whether a train/test split is performed, default: True
    # seed: Seed for the train/test split
    # antibiotics_variables: Strings containing the Antibiotic variables
    # binary_antibiotics: Boolean, whether to model the Antibiotics binary based on prescriptions or model the administrations itself, default: True
    # static_bin_ant: Boolean, whether the administration time of the Antibiotics before Sepsis onset should be considered or not, default: True
    
    #returning: 
    # final: Tensor containing all dynamic variables (including time on the first channel, antibiotics, and missing masks) for OptAB, size: batch x timepoints x variables
    # key_dict: Dictionary mapping time index to time in hours (here 1 index = 1 hour, therefore not necessary)
    # variables_: List containing all labels of the dynamic variables, which are standardized (variables, without time and missing masks)
    # variables_complete: List containing all labels of the dynamic variables
    # complete_stay_list: List of stay indices for the patients
    # tensor_static: Tensor containing all static variables for OptAB, size: batch x variables
    # static_variables: List containing all labels of the static variables
    # fit: Fitted knn_imputer object 
    # variables_mean and variables_std: Lists containing all means and standard deviations used for standardization of the dynamic variables
    # static_mean and static_std: Lists containing all means and standard deviations used for standardization of the static variables
    # indices_train and indices_test: Lists containing the training and test indices of the data

    
    tensor_list=[]
    list_nottransformed=[]
    complete_stay_list=[]
    static_ant_list=[]
    
    #preprocessing for one patient using one_patient to tensor function
    if list_of_hadms is not None:
        for i in list_of_hadms:
            if print_:
                print(i)
            te,_,s = one_patient_to_tensor(lab_df, thresh=thresh, sep_data=sep_data, index=None, hadm_id=i, round_time=round_time, round_nearest=round_nearest, round_minutes=round_minutes, aggregate_startvalues=aggregate_startvalues, start_hours=start_hours, remove_unaggregated_values=remove_unaggregated_values, max_time=max_pred, start_timepoint=start_timepoint, lab_demo=lab_demo, variables=variables, just_icu=just_icu,icustays=icustays, stay=i, antibiotics=antibiotics, remove_noant=remove_noant,antibiotics_variables=antibiotics_variables,binary_antibiotics=binary_antibiotics,static_bin_ant=static_bin_ant)
            if te is not None:
                variables_ = list(_)
                variables_complete = list(te.keys())
                tensor_list.append(torch.from_numpy(te.to_numpy().astype(np.float32)))
                list_nottransformed.append(te.to_numpy().astype(np.float32))
            if s is not None:
                static_ant_list.append(s)
        complete_stay_list=stays_list
    elif first_adms is not None:
        for i in range(first_adms):
            if print_:
                print(i)
            te,_,s = one_patient_to_tensor(lab_df, thresh=thresh, sep_data=sep_data, index=i, round_time=round_time, round_nearest=round_nearest, round_minutes=round_minutes, aggregate_startvalues=aggregate_startvalues, start_hours=start_hours, remove_unaggregated_values=remove_unaggregated_values, max_time=max_pred, start_timepoint=start_timepoint, lab_demo=lab_demo, variables=variables, just_icu=just_icu,icustays=icustays, stay=i, antibiotics=antibiotics, remove_noant=remove_noant,antibiotics_variables=antibiotics_variables,binary_antibiotics=binary_antibiotics,static_bin_ant=static_bin_ant)
            if te is not None:
                variables_ = list(_)
                variables_complete = list(te.keys())
                tensor_list.append(torch.from_numpy(te.to_numpy().astype(np.float32)))
                list_nottransformed.append(te.to_numpy().astype(np.float32))
            if s is not None:
                static_ant_list.append(s)
        complete_stay_list=stays_list
    elif stays_list is not None:
        for i in stays_list:
            if print_:
                print(i)
            te,_,s = one_patient_to_tensor(lab_df, thresh=thresh, sep_data=sep_data, index=None,hadm_id=None, round_time=round_time, round_nearest=round_nearest, round_minutes=round_minutes, aggregate_startvalues=aggregate_startvalues, remove_unaggregated_values=remove_unaggregated_values, max_time=max_pred, start_hours=start_hours, start_timepoint=start_timepoint, lab_demo=lab_demo, variables=variables, just_icu=just_icu,icustays=icustays, stay=i, antibiotics=antibiotics, remove_noant=remove_noant,antibiotics_variables=antibiotics_variables,binary_antibiotics=binary_antibiotics,static_bin_ant=static_bin_ant)
            if te is not None:
                variables_ = list(_)
                variables_complete = list(te.keys())
                tensor_list.append(torch.from_numpy(te.to_numpy().astype(np.float32)))
                list_nottransformed.append(te.to_numpy().astype(np.float32))
                complete_stay_list.append(i)
            if s is not None:
                static_ant_list.append(s)
    else:
        complete_hadms=np.sort(sep_data["encounter_id"].unique().astype(int))
        for i in complete_hadms:
            if print_:
                print(i)
            list_nottransformed.append(te.to_numpy().astype(np.float32))
            if te is not None:
                variables_ = list(_)
                variables_complete = list(te.keys())
                tensor_list.append(torch.from_numpy(te.to_numpy().astype(np.float32)))
                list_nottransformed.append(te.to_numpy().astype(np.float32))
        complete_stay_list=stays_list
    
    #preprocessing of static variables with the prep static var function
    if static is not None:
        static_list, static_keys = prep_static_var(static=static,stays_list=complete_stay_list,static_ant_list=static_ant_list,antibiotics_variables=antibiotics_variables,missing_mask=missing_mask)
    
    #missing value imputation
    if missing_imputation_start:
        from sklearn.impute import KNNImputer
        first_tp = [i[0,1:len(variables_)+1] for i in list_nottransformed]
        first_tp=pd.DataFrame(first_tp,columns=[variables_complete[1:len(variables_)+1]])
        if static is not None:
            static_df = pd.DataFrame(static_list)
            first_tp[list(static_keys)] = static_df
        imputer = KNNImputer()
        if train_test_split:
            np.random.seed(seed)
            indices_train = np.random.choice(a=list(range(first_tp.shape[0])),size=int(first_tp.shape[0]*0.8),replace=False)
            indices_test = [i for i in list(range(first_tp.shape[0])) if i not in indices_train]
            fit = imputer.fit(first_tp.iloc[indices_train])
        else:
            indices_train=None
            indices_test=None            
            fit = imputer.fit(first_tp)
        first_tp2 = fit.transform(first_tp)

        tensor_list=[]
        tensor_static_list=[]
        for i in range(len(list_nottransformed)):
            list_nottransformed[i][0,1:len(variables_)+1]=first_tp2[i,:len(variables_)]
            tensor_list.append(torch.from_numpy(list_nottransformed[i]))
            if static is not None:
                tensor_static_list.append(torch.from_numpy(first_tp2[i,len(variables_):])) 
    else:
        fit=None
            
    max_time = int(max([max(i[:,0]) for i in tensor_list if len(i)>0]))
    min_time = int(min([min(i[:,0]) for i in tensor_list if len(i)>0]))
    key_dict = {}
    
    if min_pred is None:
        min_pred = np.nan
    if max_pred is None:
        max_pred = np.nan
        
    min_pred = np.nanmax([min_pred,min_time])
    max_pred = np.nanmin([max_pred,max_time])
    
    first=True
    #Iteratively creation of tensor for all patients via for loop
    for j in np.arange(min_time-pred_times_in_h,max_time,pred_times_in_h): #min_time -1 to include the first timestamp at the min_time
        tensor_list_j=[]
        if ([i[torch.logical_and(i[:,0]<=j+pred_times_in_h,i[:,0]>j)].size(0) for i in tensor_list] and max([i[torch.logical_and(i[:,0]<=j+pred_times_in_h,i[:,0]>j)].size(0) for i in tensor_list])>0):
            max_length=max([i[torch.logical_and(i[:,0]<=j+pred_times_in_h,i[:,0]>j)].size(0) for i in tensor_list])
            for k in range(len(tensor_list)):
                if len(tensor_list[k][torch.logical_and(tensor_list[k][:,0]<=j+pred_times_in_h,tensor_list[k][:,0]>j)]):
                    te=fill_forward(tensor_list[k][torch.logical_and(tensor_list[k][:,0]<=j+pred_times_in_h,tensor_list[k][:,0]>j)],max_length)
                else:
                    if first:
                        te=torch.full((max_length, tensor_list[k][torch.logical_and(tensor_list[k][:,0]<=j+pred_times_in_h,tensor_list[k][:,0]>j)].shape[1]), torch.nan)
                        te[:,0] = min_time
                        te[:,-len(variables_):]=0
                    else: #error no problem because final is defined in the if clause some lines later
                        te=final[k,-1:,:]
                        if max_length>1:
                            te=te.repeat((max_length,1))
                tensor_list_j.append(te)
            
            tensor_j = torch.stack(tensor_list_j)
            if not first:
                final = torch.cat((final,tensor_j),dim=1)
            elif first and tensor_list_j:
                final=tensor_j
                first=False
            elif first and not tensor_list_j:
                final=[]
        elif first and not tensor_list_j:
            final=[]
        if j+pred_times_in_h>min_pred and j+pred_times_in_h<=max_pred and type(final) is not list:
            te=torch.empty(size=(final.shape[0],1,final.shape[2]))
            for k in range(final.shape[0]):
                if final[k,-1,0]==j+pred_times_in_h:
                    te[k]=final[k,-1,:]
                else:
                    help_tens=final[k,-1,:].clone()
                    help_tens[1:len(variables_)+1]=np.nan
                    help_tens[0]=j+pred_times_in_h
                    te[k]=help_tens
            if round_minutes is not None and max_length==int((60/round_minutes)*pred_times_in_h):
                final[:,-1:,:] = te
            else:
                final = torch.cat((final,te),dim=1)
        if type(final) is not list:
            key_dict[j+pred_times_in_h] = final.shape[1]-1
    
    #return of the class is based on static variables
    if static is not None:
        tensor_static=torch.stack(tensor_static_list)
        if standardize:
            if train_test_split:
                static_mean = tensor_static[indices_train].nanmean(dim=[0])[[i != 'male' and 'mask' not in i for i in static_keys]]
                static_std = np.nanstd(tensor_static[indices_train].numpy()[:,[i != 'male' and 'mask' not in i for i in static_keys]],axis=tuple([0]))
            else:
                static_mean = tensor_static.nanmean(dim=[0])[:-1]
                static_std = np.nanstd(tensor_static.numpy()[:,[i != 'male' and 'mask' not in i for i in static_keys]],axis=tuple([0]))
            tensor_static[:,[i != 'male' and 'mask' not in i for i in static_keys]] = (tensor_static[:,[i != 'male' and 'mask' not in i for i in static_keys]]-static_mean)/static_std
    
        if static_time:
            tensor_static=torch.cat([tensor_static,torch.ones(size=[tensor_static.shape[0],tensor_static.shape[1]-sum(['stat' in i for i in static_keys])])],axis=1)[:,None,:]
            a=torch.empty(size=[final.shape[0],final.shape[1]-1,tensor_static.shape[2]])
            a[:,:,:int(a.shape[2]/2)]=np.nan
            a[:,:,int(a.shape[2]/2):]=1
            b=torch.cat([tensor_static,a],axis=1)
            final=torch.cat([final,b],axis=2)
            variables_complete = variables_complete + list(static_keys)
            
        if standardize:    
            #just for next lines
            if antibiotics_variables is None:
                antibiotics_variables=[0]
            if train_test_split:
                variables_mean = final[indices_train].nanmean(dim=[0,1])[1:len(variables_)+1-len(antibiotics_variables)]
                variables_std = np.nanstd(final[indices_train].numpy()[:,:,1:len(variables_)+1-len(antibiotics_variables)],axis=tuple([0,1]))
            else:
                variables_mean = final.nanmean(dim=[0,1])[1:len(variables_)+1-len(antibiotics_variables)]
                variables_std = np.nanstd(final.numpy()[:,:,1:len(variables_)+1-len(antibiotics_variables)],axis=tuple([0,1]))
            final[:,:,1:len(variables_)+1-len(antibiotics_variables)]=(final[:,:,1:len(variables_)+1-len(antibiotics_variables)]-variables_mean)/variables_std
            
        return final, key_dict, variables_, variables_complete, complete_stay_list, tensor_static, list(static_keys), fit, variables_mean, variables_std, static_mean, static_std, indices_train, indices_test
    else:
        if standardize:    
            if antibiotics_variables is None:
                antibiotics_variables=[0]
            if train_test_split:
                variables_mean = final[indices_train].nanmean(dim=[0,1])[1:len(variables_)+1-len(antibiotics_variables)]
                variables_std = np.nanstd(final[indices_train].numpy()[:,:,1:len(variables_)+1-len(antibiotics_variables)],axis=tuple([0,1]))
            else:
                variables_mean = final.nanmean(dim=[0,1])[1:len(variables_)+1-len(antibiotics_variables)]
                variables_std = np.nanstd(final.numpy()[:,:,1:len(variables_)+1-len(antibiotics_variables)],axis=tuple([0,1]))
            
        return final, key_dict, variables_, variables_complete, complete_stay_list, fit, variables_mean, variables_std, indices_train, indices_test


def prep_static_var(static=None,stays_list=None,static_ant_list=None,antibiotics_variables=None,missing_mask=None, static_ind = None):
    # helper function for preprocessing of static variables
    
    static_list=[]
    j=0
    
    if static_ind==None:
        static_ind = ["admission_age","height","weight","male"]
    
    for i in stays_list:
        static[static["stay_id"]==i[0]]
        dat=static[static["stay_id"]==i[0]].iloc[0][static_ind].to_numpy().astype(np.float32)
        if missing_mask:
            dat=np.concatenate([dat,np.isnan(dat)])
        if static_ant_list is not None and len(static_ant_list)>0:
            dat=np.concatenate([dat,static_ant_list[j]])
            j=j+1
        static_list.append(dat)
    if missing_mask is not None:
        a=[s + "_mask" for s in static_ind]
        static_ind=static_ind+a
    if antibiotics_variables is not None and static_ant_list is not None and len(static_ant_list)>0:
        antibiotics_variables = [str(j)+'stat' for j in antibiotics_variables]
        static_ind = static_ind + antibiotics_variables
    return static_list, static_ind

def multiple_patients_creatinine_preproc(lab_df, start_hours=None, timetype="time_measured", start_timepoint='sepsis', sep_data=None, lab_demo=None,just_icu=False, icustays=None, stays_list=None):
    
    # lab_comp: DataFrame containing all covariables
    # start_hours: Time duration for considering values before the starting timepoint, if aggregation is done
    # timetype: Should be time_measured
    # start_timepoint: Different options for selecting data start time - "sepsis_icu": Latest of first measured SOFA-Score at ICU and Sepsis onset, 'icu': First measured SOFA-Score, 'sepsis': Sepsis onset, 'admission': All data
    # sep_data: DataFrame including SOFA-Scores and Sepsis onset
    # lab_demo: DataFrame including static variables
    # just_icu: True if only data measured at ICU is considered
    # icustays: Tensor including detailed information on ICU stay as provided by OpenSep
    # stays_list: List of considered stay ids (alternatively list_of_hadms can be specified)
    
    # returning list of dataframes of creatinine values for all patients
    
    lab_df = lab_df[lab_df["label"].isin(['creatinine','SOFA'])]
    crea_list=[]
    if stays_list is not None:
        for i in stays_list:
            #function for preprocessing one patient
            microbiol_prep = one_patient_crea(lab_df, hadm_id=None, start_hours=start_hours, timetype=timetype, start_timepoint=start_timepoint, sep_data=sep_data, lab_demo=lab_demo, just_icu=just_icu, icustays=icustays,stay=i)
            crea_list.append(microbiol_prep)
    return crea_list

def one_patient_crea(lab_df, hadm_id=None, start_hours=None, timetype="time_measured", start_timepoint='sepsis', sep_data=None, lab_demo=None, just_icu=False, icustays=None,stay=None,round_minutes=60):

    hadm_id = icustays[icustays["stay_id"]==stay[0]]["hadm_id"].iloc[0]    
    
    #all laboratory values of one hadm_id
    lab_hadm = lab_df[lab_df["hadm_id"]==hadm_id]
    #Just three columns needed
    lab_hadm = lab_hadm[[timetype,"value","label"]]
    lab_hadm["label"]=lab_hadm["label"].astype("string") 
    #lab_hadm = lab_hadm[lab_hadm["label"]=='SOFA']
    
    lab_hadm = lab_hadm.drop_duplicates()
    #Sorting of labvalues
    lab_hadm=lab_hadm.sort_values(by=timetype)

    if just_icu:
        if stay is not None:
            icu_hadm=icustays[icustays["stay_id"].isin(stay)]
        else:
            icu_hadm=icustays[icustays["hadm_id"]==hadm_id]

        lab_hadm=lab_hadm[(lab_hadm["time_measured"]>(min(icu_hadm["icu_intime"])-np.timedelta64(int(start_hours*60),'m'))) & (lab_hadm["time_measured"]<max(icu_hadm["icu_outtime"]))]# & (lab_hadm["time_measured"]<max(icu_hadm["dischtime"]))]
    
    if start_timepoint=='sepsis':
        start = pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0]
    elif start_timepoint=='sepsis_icu':
        start = max(pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0],min(lab_hadm[lab_hadm["label"]=='SOFA']["time_measured"]))
    elif start_timepoint=="icu":
        if stay is None:
            start = min(lab_demo["icu_intime"][lab_demo["hadm_id"]==hadm_id])
        else:
            start = min(icu_hadm["icu_intime"])
    elif start_timepoint=='admission':
        start = min(lab_demo["admittime"][lab_demo["hadm_id"]==hadm_id])
        
    elif start_timepoint=='sofa':
        start=min(lab_hadm[lab_hadm["label"]=='SOFA']["time_measured"])
        
    lab_hadm[timetype]=lab_hadm[timetype]-start
    lab_hadm = lab_hadm[lab_hadm["label"]=='creatinine']
    lab_hadm["time_measured"] = lab_hadm["time_measured"].map(to_hours)
    lab_hadm["time_measured"] = lab_hadm["time_measured"].apply(lambda x: round_nearest_func(x, a=round_minutes/60))
    
    lab_hadm=lab_hadm.pivot_table(index=timetype,columns="label",values="value")

    lab_hadm = lab_hadm.reset_index()
    
    return lab_hadm

def multiple_patients_microbiol_preproc(microbiol, lab_df, start_hours=None, timetype="time_measured", start_timepoint='sepsis', sep_data=None, lab_demo=None,just_icu=False, icustays=None, stays_list=None):
    
    # microbiol: Preprocessed microbial data from the preprocessing skript
    # start_hours: Time duration for considering values before the starting timepoint, if aggregation is done
    # timetype: Should be time_measured
    # start_timepoint: Different options for selecting data start time - "sepsis_icu": Latest of first measured SOFA-Score at ICU and Sepsis onset, 'icu': First measured SOFA-Score, 'sepsis': Sepsis onset, 'admission': All data
    # sep_data: DataFrame including SOFA-Scores and Sepsis onset
    # lab_demo: DataFrame including static variables
    # just_icu: True if only data measured at ICU is considered
    # icustays: Tensor including detailed information on ICU stay as provided by OpenSep
    # stays_list: List of considered stay ids (alternatively list_of_hadms can be specified)
    
    microbiol_list=[]
    if stays_list is not None:
        for i in stays_list:
            microbiol_prep = one_patient_microbiol(microbiol, lab_df, hadm_id=None, start_hours=start_hours, timetype=timetype, start_timepoint=start_timepoint, sep_data=sep_data, lab_demo=lab_demo, just_icu=just_icu, icustays=icustays,stay=i)
            microbiol_list.append(microbiol_prep)
    
    return microbiol_list

def one_patient_microbiol(microbiol, lab_df, hadm_id=None, start_hours=None, timetype="time_measured", start_timepoint='sepsis', sep_data=None, lab_demo=None, just_icu=False, icustays=None,stay=None):

    hadm_id = icustays[icustays["stay_id"]==stay[0]]["hadm_id"].iloc[0]    
    subject_id = icustays[icustays["stay_id"]==stay[0]]["subject_id"].iloc[0] 
    
    #all laboratory values of one hadm_id
    lab_hadm = lab_df[lab_df["hadm_id"]==hadm_id]
    #Just three columns needed
    lab_hadm = lab_hadm[[timetype,"value","label"]]
    lab_hadm["label"]=lab_hadm["label"].astype("string") 
    lab_hadm = lab_hadm[lab_hadm["label"]=='SOFA']
    
    lab_hadm = lab_hadm.drop_duplicates()
    #Sorting of labvalues
    lab_hadm=lab_hadm.sort_values(by=timetype)

    if just_icu:
        if stay is not None:
            icu_hadm=icustays[icustays["stay_id"].isin(stay)]
        else:
            icu_hadm=icustays[icustays["hadm_id"]==hadm_id]

        lab_hadm=lab_hadm[(lab_hadm["time_measured"]>(min(icu_hadm["icu_intime"])-np.timedelta64(int(start_hours*60),'m'))) & (lab_hadm["time_measured"]<max(icu_hadm["icu_outtime"]))]# & (lab_hadm["time_measured"]<max(icu_hadm["dischtime"]))]
    
    if start_timepoint=='sepsis':
        start = pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0]
    elif start_timepoint=='sepsis_icu':
        start = max(pd.to_datetime(sep_data[sep_data["encounter_id"].astype("int64")==hadm_id]["Sepsis_Time"].values,utc=False)[0],min(lab_hadm[lab_hadm["label"]=='SOFA']["time_measured"]))
    elif start_timepoint=="icu":
        if stay is None:
            start = min(lab_demo["icu_intime"][lab_demo["hadm_id"]==hadm_id])
        else:
            start = min(icu_hadm["icu_intime"])
    elif start_timepoint=='admission':
        start = min(lab_demo["admittime"][lab_demo["hadm_id"]==hadm_id])
    lab_hadm[timetype]=lab_hadm[timetype]-start
    
    microbiol_id = microbiol[microbiol["subject_id"]==subject_id]
    
    microbiol_id["charttime"] = microbiol_id["charttime"]-start
    microbiol_id["storetime"] = microbiol_id["storetime"]-start
    microbiol_id["chartdate"] = microbiol_id["chartdate"]-start
    
    microbiol_id["charttime"] = microbiol_id["charttime"].map(to_hours)
    microbiol_id["storetime"] = microbiol_id["storetime"].map(to_hours)    
    microbiol_id["chartdate"] = microbiol_id["chartdate"].map(to_hours) 
    
    return microbiol_id