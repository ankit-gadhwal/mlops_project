import numpy as np
import pandas as pd
import os
def load_data(filepath:str) ->pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error in data loading {filepath} : {e}")
# train_data = pd.read_csv(r"data\raw\train_csv")
# test_data = pd.read_csv(r"data\raw\test_csv")

def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value,inplace = True)
    return df

def save_Data(df : pd.DataFrame,filepath : str) ->None:
    try:
        df.to_csv(filepath,index = False)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath} : {e}")
# train_processed_data = fill_missing_with_median(train_data)
# test_processed_data =fill_missing_with_median(test_data)
def main():
    try:
        raw_data_path = "data/raw"
        processed_data_path = "data/processed"
        
        train_path = os.path.join(raw_data_path, "train.csv")
        print("TRAIN PATH:", train_path)

       
        train_data = load_data(os.path.join(raw_data_path, "train_csv"))
        test_data = load_data(os.path.join(raw_data_path, "test_csv"))

        train_processed_data = fill_missing_with_mean(train_data)
        test_processed_data =  fill_missing_with_mean(test_data)
# data_path = os.path.join("data","processed")
    
        os.makedirs(processed_data_path,exist_ok=True)
    
        save_Data(train_processed_data,os.path.join(processed_data_path,"train_processed_median.csv"))
        save_Data(test_processed_data,os.path.join(processed_data_path,"test_processed_median.csv"))
    except Exception as e:
        raise Exception(f"An error occured : {e}")
if __name__ == "__main__":
    main()    
# train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index = False)
# test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index = False)
# data\raw\train_csv