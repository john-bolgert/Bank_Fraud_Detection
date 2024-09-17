from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import pandas as pd
import numpy as np
import random

def df_metadata(df): 
    #function returns datatypes from dataframe as either categorical or numerical 
    #this is used as a check to ensure the metadata information passed to data generator is correct 
    df_types = df.dtypes.to_dict() #create dictionary with datatypes
    
    # Correct For Ordinal vars stored as numeric vars
    cols  = list(df.columns)
    numeric_cats = []
    for col in cols:
        if len(df[col].unique())<7:
            numeric_cats.append(col)
            
    for val in df_types: #loop through values 
        if val == 'fraud_bool' or val in numeric_cats: 
            df_types[val]= 'categorical'
        elif df_types[val] == 'int64' or df_types[val] == 'float64': 
            df_types[val] = 'numerical' #if int set to numerical
        else: 
            df_types[val]= 'categorical' #else categorical 
    return df_types

def syn_metadata_check(metadata, df): 
    #function requires input of metadata and dataframe 
    #function will then update the metadata for any mismatches 
    df_type_dict = df_metadata(df) #call df_metadata to get dictionary of values 
    syn_type_dict = metadata.to_dict()#generate metadata from svd 
    col_lst = df.columns.values.tolist() #create list of variables 
    for val in col_lst: #iterate through variables 
        if df_type_dict[val] != syn_type_dict['columns'][val]['sdtype']: #check if types dont match
            metadata.update_column(
                column_name=val,
                sdtype=df_type_dict[val] #update type 
                )

def df_add_noise(df_syn,magnitude,outcome_val,non_fraud_df): 
    #function requires input of dataframe, magintude value and the variable one is trying to predict
    #it will return a new dataframe with added noise to the inputted dataset 
    num_frd = len(df_syn)
    df_types = df_metadata(df_syn) #generate metadata from dataframe
    df_syn_noise = df_syn.copy() #create copy of dataframe
    
    # Correct For Ordinal vars stored as numeric vars
    cols  = list(df_syn.columns)
    numeric_cats = []
    for col in cols:
        if len(df_syn[col].unique())<7:
            numeric_cats.append(col)
    
    for val in df_types: #iterate through list of types
        if val != outcome_val: #make sure we are not updating value we are trying to predict
            if df_types[val] == 'numerical' and val not in numeric_cats: #if numerical 
                
                # get 20% quartiles from non fraud df
                qbins = pd.qcut(non_fraud_df[val],5,retbins=True, duplicates='drop')[1]
                
                #generate new distribution
                dist_pct = []
                
                for i in range(len(qbins)-1):
                    add = random.uniform(1, 1+magnitude)
                    
                    dist_pct.append(add)
                dist_pct = [lsnum /sum(dist_pct) for lsnum in dist_pct]
                dist_pct = [round(lsnum * num_frd) for lsnum in dist_pct]
                if sum(dist_pct)<num_frd:
                    center = round(len(dist_pct)/2)
                    lower = center-1
                    upper = center+1
                    add_ind = int(random.uniform(lower, upper))
                    dist_pct[add_ind] = dist_pct[add_ind]+(num_frd-sum(dist_pct))
                    
                if sum(dist_pct)>num_frd:
                    center = round(len(dist_pct)/2)
                    lower = center-1
                    upper = center+1
                    add_ind = int(random.uniform(lower, upper))
                    dist_pct[add_ind] = dist_pct[add_ind]-(sum(dist_pct)-num_frd)
                
                new_fraud_vals = []
                for i in range(len(qbins)-1):
                    min_val = qbins[i]
                    max_val = qbins[i+1]
                    for new in range(dist_pct[i]):
                        new_num = random.uniform(min_val, max_val)
                        new_fraud_vals.append(new_num)
                
                
                random.shuffle(new_fraud_vals)
                new_fraud_vals = new_fraud_vals[:num_frd]
                df_syn_noise[val] = new_fraud_vals
                # old code
                #std =  df_syn_noise[val].std() * (1 + magnitude) #offset std
                #df_syn_noise[val] = df_syn_noise[val].add(np.random.normal(0, std, df_syn_noise.shape[0])) #generate random value based on increased std and add to prexisting values in respective column
            else: #else categorical variable
                uniq_vals = non_fraud_df[val].unique()
                dist_pct = []
                for i in range(len(uniq_vals)-1):
                    add = random.uniform(1, 1+magnitude)
                    dist_pct.append(add)
                dist_pct = [lsnum /sum(dist_pct) for lsnum in dist_pct]
                dist_pct = [round(lsnum * num_frd) for lsnum in dist_pct]
                
                
                new_fraud_vals = []
                for i in range(len(uniq_vals)-1):
                    add_val = uniq_vals[i]
                    for new in range(dist_pct[i]):
                        new_fraud_vals.append(add_val)
                        
                new_fraud_vals = new_fraud_vals[:num_frd]
                if len(new_fraud_vals)<num_frd:
                    for add in range(num_frd-len(new_fraud_vals)):
                        new_fraud_vals.append(add_val)
                random.shuffle(new_fraud_vals)
                df_syn_noise[val] = new_fraud_vals #if value calcualted is greater than magnitude then replace value 
    
    return df_syn_noise #return new dataframe 

# This function will create new data set by taking base data a df bool value and magnitude of the drift
def create_new_dataset (base_data,n_rows,outcome_percentage,outcome_val,df_min_max, concept_drift=False,drift_magnitude=None):
    #dataset will create new dataset with the required inputs of base dataframe, number of rows needed to be generated, the percentage of the rows the value of interest is found 
    #Additional parmaters include, concept_drift and drift_magnitude, if no values are passed for these values they are set to False and None respectively. 
    #If values are passed as True then noise will be added to the generated data based on the maginitude passed in. 
    
    org_cols = list(base_data.columns)
    metadata = SingleTableMetadata() #call generator function 
    metadata.detect_from_dataframe(base_data) #generate metadata
    syn_metadata_check(metadata,base_data) #call syn_metadata_check() to update metadata if needed. 
    
    #metadata.validate_data(data=base_data)

    synthesizer = GaussianCopulaSynthesizer(metadata, 
                                        enforce_min_max_values=True,
                                        enforce_rounding=True) #create synthesizer
    synthesizer_fraud = GaussianCopulaSynthesizer(metadata, 
                                        #enforce_min_max_values=True,
                                        enforce_rounding=True) #create synthesizer 
    
    df_o = base_data[base_data[outcome_val]==1] #dataframe of outcome variables of interest
    n_rows_o = int(n_rows * outcome_percentage) #determine number of rows needed for outcome variable of interest
    df_no = base_data[base_data[outcome_val]==0] #dataframe for non outcome variable of interest
    n_rows_no = n_rows - n_rows_o  #determine number of rows needed 
    
    synthesizer_fraud.fit(df_o) #fit model on data 
    syn_data_o = synthesizer_fraud.sample(num_rows=n_rows_o) #synethsize data for where outcome variable of interest is present 
    
    synthesizer.fit(df_no) #fit model on data 
    syn_data_no = synthesizer.sample(n_rows_no)#synethsize data for where outcome variable of interest is not present 

    #Set the new Min/Max of sythn data to the Min/Max of Orginal Data
    for col in org_cols:
        if syn_data_no[col].dtype in ['int64', 'float64']:
            if col!='fraud_bool':
                #Reset Max Val
                ind = list(syn_data_no[syn_data_no[col] == syn_data_no[col].max()].index)[0]
                syn_data_no.loc[ind,col] = df_min_max.loc['Max',col]
                
                #Reset Min Val
                ind = list(syn_data_no[syn_data_no[col] == syn_data_no[col].min()].index)[0]
                syn_data_no.loc[ind,col] = df_min_max.loc['Min',col]
                
    
    if concept_drift != False: #if concept drift is not false 
        syn_data_o = df_add_noise(syn_data_o,drift_magnitude,outcome_val,syn_data_no) #add noise to data generated where outcome variable of interest is present 
        
    
    return pd.concat([syn_data_o, syn_data_no], ignore_index=True) #combine data 
        