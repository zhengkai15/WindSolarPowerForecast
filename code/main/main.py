
import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
# import logging
from loguru import logger
from typing import List, Dict, Union, Tuple
import pickle
from datetime import timedelta
import matplotlib.pyplot as plt
import math
from common.utils import *

             
def train(farm_id, model_name, version='v1',power_type="ws"):
    x_df = pd.DataFrame()
    nwp_train_path = f'{base_path}/TRAIN/nwp_data_train/{farm_id}'
    catch_dir = f"{base_path}/catch_dir_all_{grid_num}_{freq}/TRAIN"
    # catch_dir = f"{base_path}/catch_dir_all_1h/TRAIN"  # TODO 读取1h 插值到15min
    os.makedirs(catch_dir,exist_ok=True)
    nwp_csv_filename  = os.path.join(catch_dir, f"nwp_catch_{farm_id}.csv") 
    try:
        x_df = pd.read_csv(nwp_csv_filename, index_col = 0, parse_dates=[0])
        # x_df = pd.read_csv(nwp_csv_filename, index_col = 0, parse_dates=[0]).resample('15min').interpolate(method='linear') # TODO 读取1h 插值到15min
    except:
        for nwp in nwps:
            nwp_path = os.path.join(nwp_train_path, nwp)
            x_df = process_nwp_data(nwp_path, nwp, x_df, days_num=365, day_point_num=day_point_num)
        x_df.index = pd.date_range(datetime(2024, 1, 2, 0, 0), datetime(2024, 12, 31, 23, 59), freq=freq)
        x_df.to_csv(nwp_csv_filename)
    
    x_df = apply_feature_engineering(x_df, version=version, power_type=power_type, farm_id=farm_id)
    
    y_df = pd.read_csv(os.path.join(fact_path,f'{farm_id}_normalization_train.csv'),index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']
    
    if power_type == "ws":
        flag = "NWP_1_ws_5"
    elif  power_type == "pv":
        flag = "NWP_1_poai_5"
        
    x_processed,y_processed = data_preprocess(x_df,y_df,flag=flag,farm_id=farm_id)
    
    y_processed[y_processed < 0] = 0 # clip minus to 0
    
    # xgb 
    model = get_model(model_name=model_name)
    model.fit(x_processed,y_processed)
    
    return model


def predict(model,farm_id, version="v1", power_type="ws", flag="train"):
    x_df = pd.DataFrame()
    nwp_test_path = f'{base_path}/TEST/nwp_data_{flag}/{farm_id}'
    catch_dir = f"{base_path}/catch_dir_all_{grid_num}_{freq}/{flag.upper()}"
    os.makedirs(catch_dir,exist_ok=True)
    nwp_csv_filename  = os.path.join(catch_dir, f"nwp_catch_{farm_id}.csv") 
    try:
        x_df = pd.read_csv(nwp_csv_filename, index_col = 0, parse_dates=[0])
    except:
        for nwp in nwps:
            nwp_path = os.path.join(nwp_test_path, nwp)
            x_df = process_nwp_data(nwp_path, nwp, x_df, days_num=59, day_point_num=day_point_num)
        if flag == "test":
            x_df.index = pd.date_range(datetime(2025, 1, 1, 0, 0), datetime(2025, 2, 28, 23, 59), freq=freq) # 对齐数据的之间范围               
        elif flag == "train":
            x_df.index = pd.date_range(datetime(2024, 1, 2, 0, 0), datetime(2024, 12, 31, 23, 59), freq=freq)
        x_df.to_csv(nwp_csv_filename)
    x_df_tmp = x_df.copy()
    x_df = apply_feature_engineering(x_df, version=version, power_type=power_type, farm_id=farm_id)
    
    pred_pw = model.predict(x_df).flatten() 
    import warnings
    warnings.filterwarnings("ignore")
    if power_type == "ws":
        parm = 0
        logger.info(f"建模{power_type}, 后处理:-{parm}")
        pred_pw = pred_pw - parm
    elif power_type == "pv":
        parm = 0
        logger.info(f"建模{power_type}, 后处理:-{parm}")
        pred_pw = pred_pw - parm
        # 后处理
        pred_pw[x_df_tmp["NWP_1_ghi_0"]==0] = 0        
    else:
        raise ValueError("Unsupported power type")
    
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0],periods=len(pred_pw), freq=freq))
    res = pred.resample('15min').interpolate(method='linear') 
    res[res<0] = 0
    res[res>1] = 1
    return res

if __name__ == "__main__":
    acc = pd.DataFrame()
    farms = {1: 'ws', 2: 'ws', 3: 'ws', 4: 'ws', 5: 'ws', 6: 'pv', 7: 'pv', 8: 'pv', 9: 'pv', 10: 'pv'}
    
    # 试验路径
    from tqdm import tqdm
    
    for farm_id in tqdm(farms.keys()):
        power_type = farms[farm_id]
        model_name = "xgb" # 模型版本: xgb XGBoostEnsemble
        exp_dir = f"../../output/{model_name}"
        version_ft = "v5_ws_and_pv_add_1-9_add_tp" # 特征版本 v5_ws_and_pv, v5_ws_and_pv_add_1-9 v5_ws_and_pv_add_1-9_add_t2m  v5_ws_and_pv_add_1-9_add_tp v5_ws_and_pv_add_1-9_add_tp_wd, v5_ws_and_pv_add_1-9_add_tp_add_loc
        version = f"{model_name}_{version_ft}"
        logger.info(f"建模{power_type}, version:{version}")
        model_path = f'{exp_dir}/models/{farm_id}'
        os.makedirs(model_path,exist_ok=True)
        logger.info(f"训练...")
        model = train(farm_id, model_name=model_name, version=version_ft, power_type=power_type)
        
        logger.info(f"推理...")

        for flag in ["train","test"]:
            pred = predict(model,farm_id, version=version_ft, power_type=power_type,  flag=flag)
            result_path = f'{exp_dir}/results/output_{flag}'
            os.makedirs(result_path,exist_ok=True)
            pred.to_csv(os.path.join(result_path,f'output{farm_id}.csv'))
        
    
    import zipfile
    zip_file_name = os.path.join(os.path.dirname(result_path),f'output_{version}.zip')
    print(zip_file_name)
    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(result_path):
            for file in files:
                print(os.path.join(root, file))
                print(os.path.relpath(os.path.join(root, file), os.path.join(result_path, '..')))
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(result_path, '..')))
    
    print('****DONE****')
