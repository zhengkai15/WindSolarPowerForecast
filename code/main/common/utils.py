import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from loguru import logger
import pickle
from datetime import timedelta
import math
# 设置显示选项
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_seq_items', None)

nwps = ['NWP_1','NWP_2','NWP_3']
base_path = "../../data"
fact_path = f'{base_path}/TRAIN/fact_data'
freq = "1h" # "1h" 15min
grid_num = 3
if freq == "15min":
    time_ratio = 4
    day_point_num = 96 
    
elif freq == "1h":
    time_ratio = 1
    day_point_num = 24 
else:
    raise "wroing freq"

def balance_data_by_bins(x_df, y_df):
    df = pd.concat([x_df, y_df], axis=1)
    num_bins = 5
    df['bins'] = pd.cut(df['power'], bins=num_bins)
    bin_counts = df['bins'].value_counts()
    target_count = bin_counts.min()
    balanced_df = pd.DataFrame()
    for bin_val in bin_counts.index:
        bin_data = df[df['bins'] == bin_val]
        sampled_data = bin_data.sample(n=target_count, random_state=42)
        balanced_df = pd.concat([balanced_df, sampled_data])
    balanced_df = balanced_df.drop(columns=['bins'])
    y_df = balanced_df[['power']]
    x_df = balanced_df.drop(columns=['power'])
    return x_df, y_df

def balance_data_by_threshold(x_df, y_df, threshold=None):
    df = pd.concat([x_df, y_df], axis=1)
    if threshold is None:
        threshold = df['power'].median()
    small_values = df[df['power'] <= threshold]
    large_values = df[df['power'] > threshold]
    target_size = len(small_values)
    oversampled_large_values = large_values.sample(n=target_size, replace=True, random_state=42)
    balanced_df = pd.concat([small_values, oversampled_large_values])
    balanced_df = balanced_df.drop(columns=['bins'])
    y_df = balanced_df[['power']]
    x_df = balanced_df.drop(columns=['power'])
    return x_df, y_df

def data_preprocess(x_df, y_df, flag, balance_type=None, farm_id=1):
    # 数据重采样    
    x_df = x_df.dropna()
    y_df = y_df.dropna()
    
    # 数据对扣
    ind = x_df.index.intersection(y_df.index) # 数据对齐到1h
    x_df = x_df.loc[ind]
    y_df = y_df.loc[ind]
    
    print("原始数据分布：")
    print(y_df['power'].describe())
    
    # 数据预处理
    if balance_type == 'bins':
        x_df, y_df = balance_data_by_bins(x_df, y_df)
    elif balance_type == 'threshold':
        x_df, y_df = balance_data_by_threshold(x_df, y_df)
    else:
        logger.info("None balance_type Now. Choose 'bins' or 'threshold'.")

    print("\n平衡后数据分布：")
    print(y_df['power'].describe())
    
    x_df = x_df.iloc[:,:]
    y_df = y_df.iloc[:,:]

    return x_df,y_df

import warnings
warnings.filterwarnings("ignore")
import xgboost as xgb
import numpy as np

class XGBoostEnsemble:
    def __init__(self, num_models=5, params=None, random_seeds=[42,43,44,45,46]):
        """
        初始化 XGBoost 集成模型
        :param num_models: 集成的 XGBoost 模型数量
        :param params: XGBoost 模型的参数
        :param random_seeds: 每个模型的随机种子列表
        """
        
        if random_seeds is None:
            random_seeds = list(range(num_models))

        self.num_models = num_models
        self.params = params
        self.random_seeds = random_seeds
        self.models = []

    def fit(self, X, y):
        """
        训练集成中的每个 XGBoost 模型
        :param X: 特征矩阵
        :param y: 目标值
        """
        for seed in self.random_seeds:
            model = xgb.XGBRegressor(random_state=seed, **self.params)
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X):
        """
        对输入数据进行预测，并使用简单平均法集成预测结果
        :param X: 特征矩阵
        :return: 集成预测结果
        """
        predictions = []
        for model in self.models:
            y_pred = model.predict(X)
            predictions.append(y_pred)
        return np.mean(predictions, axis=0)

    
    
def get_model(model_name="xgb"):
    # 参数解释：https://www.cnblogs.com/TimVerion/p/11436001.html
    if model_name=="xgb":
        # 创建 XGBRegressor 模型，并设置参数
        model = xgb.XGBRegressor(
            objective='reg:squarederror',  # 回归任务，使用平方误差损失函数
            random_state=42, # 随机数
            max_leaves=31,  # 限制每棵树的最大叶子结点数
            max_depth=6,  # 限制树的最大深度
            learning_rate=0.05,  # 学习率
            subsample=0.8,  # 样本采样比例
            colsample_bytree=0.8,  # 特征采样比例
            n_estimators = 100, # 默认100
            # reg_alpha = 0, # 默认0, L1正则 加上默认参数结果变差
            # reg_lambda = 0, # 默认0, L2正则 加上默认参数结果变差
            gamma=0,
        )
    elif model_name == "lgb":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(
            objective='regression',  # 回归任务
            random_state=42,  # 随机数种子
            num_leaves=31,  # 限制每棵树的最大叶子结点数，对应 XGBoost 的 max_leaves
            max_depth=6,  # 限制树的最大深度
            learning_rate=0.05,  # 学习率
            subsample=0.8,  # 样本采样比例
            colsample_bytree=0.8,  # 特征采样比例，在 LightGBM 中对应 feature_fraction
            n_estimators=100,  # 迭代次数，默认 100
            reg_alpha=0,  # L1 正则化系数，默认 0
            reg_lambda=0,  # L2 正则化系数，默认 0
            min_split_gain=0  # 对应 XGBoost 的 gamma，分裂所需的最小增益
        )
        
    elif model_name == "XGBoostEnsemble":
        params = {"objective":'reg:squarederror',  # 回归任务，使用平方误差损失函数
            "max_leaves":31,  # 限制每棵树的最大叶子结点数
            "max_depth":6,  # 限制树的最大深度
            "learning_rate":0.05,  # 学习率
            "subsample":0.8,  # 样本采样比例
            "colsample_bytree":0.8,  # 特征采样比例
            "n_estimators":100, # 默认100
            "gamma":0}
        model = XGBoostEnsemble(num_models=5, params=params, random_seeds=[42,43,44,45,46])
    else:
        raise "wrong model_name"
    return model


def extract_time_features(df):
    time_features = pd.DataFrame()
    time_features['hour'] = df.index.hour
    # 小时的周期性特征
    time_features['hour_sin'] = np.sin(df.index.hour * (2 * np.pi / 24))
    time_features['hour_cos'] = np.cos(df.index.hour * (2 * np.pi / 24))
    
    time_features['day'] = df.index.day
    time_features['day_sin'] = np.cos(df.index.day * (2 * np.pi / 31))
    time_features['month'] = df.index.month
    time_features['month_sin'] = np.sin(df.index.month * (2 * np.pi / 12))
    time_features['month_cos'] = np.cos(df.index.month * (2 * np.pi / 12))
    time_features['weekday'] = df.index.weekday
    time_features['minute'] = df.index.minute
    time_features.index = df.index
    return time_features


def extract_time_features_station_3(df):
    time_features = pd.DataFrame()
    time_features['hour'] = df.index.hour
    # 小时的周期性特征
    time_features['hour_sin'] = np.sin(df.index.hour * (2 * np.pi / 24))
    time_features['hour_cos'] = np.cos(df.index.hour * (2 * np.pi / 24))
    time_features['minute'] = df.index.minute
    time_features.index = df.index
    return time_features

def extract_statistical_features(df, point_idx=[0,1,2,3,4,5,6,7,8]):
    '''
    计算空间多点的统计量
    '''
    statistical_features = pd.DataFrame()
    for i in range(1, 4):
        for var in ['u', 'v', 'ws']:
            columns = [f'NWP_{i}_{var}_{j}' for j in point_idx]
            group = df[columns]
            # 计算均值
            statistical_features[f'NWP_{i}_{var}_mean'] = group.mean(axis=1)
            # 计算标准差
            statistical_features[f'NWP_{i}_{var}_std'] = group.std(axis=1)
            # 计算最大值
            statistical_features[f'NWP_{i}_{var}_max'] = group.max(axis=1)
            # 计算最小值
            statistical_features[f'NWP_{i}_{var}_min'] = group.min(axis=1)
    return statistical_features


def extract_combined_features(df, point_idx=[0,1,2,3,4,5,6,7,8]):
    '''
    计算风向
    '''
    combined_features = pd.DataFrame()
    # 风向
    for i in range(1, 4):
        for j in point_idx:
            u_col = f'NWP_{i}_u_{j}'
            v_col = f'NWP_{i}_v_{j}'
            # 计算风向（弧度）
            combined_features[f'NWP_{i}_wd_{j}'] = np.arctan2(df[v_col], df[u_col])
    return combined_features

def calculate_sliding_features(features):
    '''
    计算时序滑窗统计量
    '''
    sliding_features = pd.DataFrame()
    for col in features.columns:
        # 计算 5 h滑动平均
        sliding_features[f'{col}_rolling_mean'] = features[col].rolling(window=5*time_ratio).mean()
        # 计算 5 h滑动最大值
        sliding_features[f'{col}_rolling_max'] = features[col].rolling(window=5*time_ratio).max()
        # 计算 5 h滑动最小值
        sliding_features[f'{col}_rolling_min'] = features[col].rolling(window=5*time_ratio).min()
    return sliding_features.bfill()

def calculate_shift_features(features):
    '''
    计算时间移动统计量
    '''
    sliding_features = pd.DataFrame()
    for col in features.columns:
        # 计算 1 h shift
        sliding_features[f'{col}_shift-1'] = features[col].shift(-1*time_ratio)
        # sliding_features[f'{col}_diff-1'] = features[col].diff(-1*time_ratio)
        # 计算 2 h shift
        sliding_features[f'{col}_shift-2'] = features[col].shift(-2*time_ratio)
        # sliding_features[f'{col}_diff-2'] = features[col].diff(-2*time_ratio)
        # 计算 3 h shift
        sliding_features[f'{col}_shift-3'] = features[col].shift(-3*time_ratio)
        # sliding_features[f'{col}_diff-3'] = features[col].diff(-3*time_ratio)
    return sliding_features.ffill()
    
def calculate_ws_log_features(features):
    '''
    计算log
    '''
    log_features = pd.DataFrame()
    for col in features.columns:
        log_features[f'{col}_log'] = np.log1p(features[col])
    return log_features.ffill()


def calculate_ws_pow_features(features):
    '''
    计算3次方
    '''
    pow_features = pd.DataFrame()
    for col in features.columns:
        # pow_features[f'{col}_pow3'] = np.power(features[col], 3)
        # pow_features[f'{col}_pow2'] = np.power(features[col], 2)
        pow_features[f'{col}_pow3'] = np.power(features[col].shift(-1).ffill(), 3)
        # pow_features[f'{col}_pow2'] = np.power(features[col].shift(-1), 2)
    return pow_features

def apply_feature_engineering(x_df, version, power_type="ws", farm_id=1):
        if version == "v1":
            # 特征工程v1 时间（没有分钟） 气象源尺度统计特征 
            statistical_features = extract_statistical_features(x_df)
            time_features = extract_time_features(x_df)
            x_df = pd.concat([x_df, statistical_features, time_features], axis=1)
        elif version == "v2":
            # 特征工程v2 # 时间 气象源尺度统计特征 气象源尺度特征5点时间尺度滑窗 风向 
            statistical_features = extract_statistical_features(x_df)
            time_features = extract_time_features(x_df)
            combined_features = extract_combined_features(x_df)
            statistical_features_slide = calculate_sliding_features(statistical_features)
            x_df = pd.concat([x_df, statistical_features, time_features, combined_features, statistical_features_slide], axis=1)
        elif version == "v3":
            # 特征工程v3 时间 气象源尺度统计特征 气象源尺度特征5点时间尺度滑窗
            statistical_features = extract_statistical_features(x_df)
            time_features = extract_time_features(x_df)
            statistical_features_slide = calculate_sliding_features(statistical_features)
            x_df = pd.concat([x_df, statistical_features, time_features, statistical_features_slide], axis=1)
        elif version == "v4":
            # 特征工程v4 时间 + 月sin 月cos 气象源尺度统计特征 气象源尺度特征5点时间尺度滑窗
            statistical_features = extract_statistical_features(x_df)
            time_features = extract_time_features(x_df)
            statistical_features_slide = calculate_sliding_features(statistical_features)
            statistical_shift_slide = calculate_shift_features(statistical_features)
            x_df = pd.concat([x_df, statistical_features, time_features, statistical_features_slide, statistical_shift_slide], axis=1)
        elif version == "v5":
            # 第五个格点的特征
            df = x_df
            df_v5 = df[[i for i in df.columns if i.endswith("_5")]]
            df_v5_ws = df[[i for i in df.columns if i.endswith("ws_5")]]
            time_features = extract_time_features(df_v5_ws)
            statistical_features_slide = calculate_sliding_features(df_v5_ws)
            statistical_features_shift = calculate_shift_features(df_v5_ws)
            x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
        elif version == "v5_log":
            # 第五个格点的特征 + log
            df = x_df
            df_v5 = df[[i for i in df.columns if i.endswith("_5")]]
            df_v5_ws = df[[i for i in df.columns if i.endswith("ws_5")]]
            time_features = extract_time_features(df_v5_ws)
            statistical_features_slide = calculate_sliding_features(df_v5_ws)
            statistical_features_shift = calculate_shift_features(df_v5_ws)
            statistical_features_log = calculate_ws_log_features(df_v5_ws)
            x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features, statistical_features_log], axis=1).dropna(axis=1)
        elif version == "v5_ws_and_pv":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws', 'u', 'v']]] # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_ws = df[[i for i in df.columns if i.endswith("ws_5")]]
                time_features = extract_time_features(df_v5_ws)
                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点
                time_features = extract_time_features(df_v5_pv)
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
        elif version == "v5_ws_and_pv_add_1-9":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['0','1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v']]] # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws']]] # 筛选变量和格点
                time_features = extract_time_features(df_v5_ws)
                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['0','1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点
                time_features = extract_time_features(df_v5_pv)
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
        elif version == "v5_ws_and_pv_add_1-9_add_t2m":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['0','1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v', 't2m']]] # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws']]] # 筛选变量和格点
                time_features = extract_time_features(df_v5_ws)
                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['0','1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ghi', 'poai','t2m']]] # # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点
                time_features = extract_time_features(df_v5_pv)
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
        elif version == "v5_ws_and_pv_add_1-9_add_tp":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v','tp']]] # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                # df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点
                df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点 
                
                time_features = extract_time_features_station_3(df_v5_ws)

                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
                
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ghi', 'poai','tp']]] # # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点 4是最中间的格点
               
                # time_features = extract_time_features(df_v5_pv)
                time_features = extract_time_features_station_3(df_v5_pv) # xgb辐照去掉年循环时间信息可以提升
                
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
        elif version == "v5_ws_and_pv_add_1-9_add_tp_add_loc":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v','tp']]] # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                # df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点
                df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点 
                
                time_features = extract_time_features(df_v5_ws)
                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                # TODO:pow
                # x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
                df_v5_ws_pow = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws']]] # 筛选变量和格点 
                statistical_features_loc = extract_statistical_features(df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v']]], point_idx=[1,2,3,4,5,6,7,8])
                statistical_features_pow = calculate_ws_pow_features(df_v5_ws_pow)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, statistical_features_pow, statistical_features_loc, time_features], axis=1).dropna(axis=1)
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ghi', 'poai','tp']]] # # 筛选变量和格点
                logger.info(f"df_v5.columns:{df_v5.columns}")
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点 4是最中间的格点
                time_features = extract_time_features(df_v5_pv)
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1) 
                
        elif version == "v5_ws_and_pv_add_1-9_add_tp_wd":
            # 第五个格点的特征 
            df = x_df
            if power_type=="ws":
                logger.info(f"{power_type}:构建风速特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws', 'u', 'v','tp']]] # 筛选变量和格点
                # df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点
                df_v5_ws = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ws','tp']]] # 筛选变量和格点
                df_v5_uv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['u', 'v']]] # 筛选变量和格点 
                df_v5_uv_wd = extract_combined_features(df_v5_uv, point_idx=['5'])
                df_v5_wd = df_v5_uv_wd[[i for i in df_v5_uv_wd.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['wd']]] # 筛选变量和格点 
                time_features = extract_time_features(df_v5_ws)
                statistical_features_slide = calculate_sliding_features(df_v5_ws)
                statistical_features_shift = calculate_shift_features(df_v5_ws)
                x_df = pd.concat([df_v5, df_v5_wd, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
                logger.info(f"df_v5.columns:{df_v5.columns}")
            elif power_type=="pv":
                logger.info(f"{power_type}:构建辐照特征")
                df_v5 = df[[i for i in df.columns if i.split("_")[-1] in ['1','2','3','4', '5', '6','7','8'] and i.split("_")[2] in ['ghi', 'poai','tp']]] # # 筛选变量和格点
                df_v5_pv = df[[i for i in df.columns if i.split("_")[-1] in ['5'] and i.split("_")[2] in ['ghi', "poai"]]] # # 筛选变量和格点 4是最中间的格点
                time_features = extract_time_features(df_v5_pv)
                statistical_features_slide = calculate_sliding_features(df_v5_pv)
                statistical_features_shift = calculate_shift_features(df_v5_pv)
                x_df = pd.concat([df_v5, statistical_features_shift, statistical_features_slide, time_features], axis=1).dropna(axis=1)
                logger.info(f"df_v5.columns:{df_v5.columns}")
        else:
            raise ValueError(f"Unsupported version: {version}")
        logger.info(f"ft num:{len(x_df.columns)},ft.col:{x_df.columns}")
        return x_df
   
   
def process_nwp_data(nwp_path, nwp, x_df, days_num, day_point_num=96):
    nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc").interp(lead_time=np.linspace(0, 24, day_point_num+1)[:-1], method="nearest") # 0-23插值 到0.15 interval 0-23.75
    # 数据中没有lat lon信息
    if grid_num == 11:
        loction_range = range(0,11)
    elif grid_num == 3:
         loction_range = range(4,7)
    loction_num = len(loction_range)**2
    # 数据中没有lat lon信息
    u = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['u100']).data.values.reshape(days_num * day_point_num, 9)
    v = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['v100']).data.values.reshape(days_num * day_point_num, 9)
    ghi = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['ghi']).data.values.reshape(days_num * day_point_num, 9)
    poai = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['poai']).data.values.reshape(days_num * day_point_num, 9)
    try:
        sp = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['sp']).data.values.reshape(days_num * day_point_num, 9)
    except:
        msl = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['msl']).data.values.reshape(days_num * day_point_num, 9)
        sp = msl
    t2m = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['t2m']).data.values.reshape(days_num * day_point_num, 9)
    tp = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['tp']).data.values.reshape(days_num * day_point_num, 9)
    tcc = nwp_data.sel(lat=range(4,7), lon=range(4,7), lead_time=np.linspace(0, 24, day_point_num+1)[:-1], channel=['tcc']).data.values.reshape(days_num * day_point_num, 9)
    
    u_df = pd.DataFrame(u, columns=[f"{nwp}_u_{i}" for i in range(u.shape[1])])
    v_df = pd.DataFrame(v, columns=[f"{nwp}_v_{i}" for i in range(v.shape[1])])
    ghi_df = pd.DataFrame(ghi, columns=[f"{nwp}_ghi_{i}" for i in range(u.shape[1])])
    poai_df = pd.DataFrame(poai, columns=[f"{nwp}_poai_{i}" for i in range(u.shape[1])])
    t2m_df = pd.DataFrame(t2m, columns=[f"{nwp}_t2m_{i}" for i in range(u.shape[1])])
    tp_df = pd.DataFrame(tp, columns=[f"{nwp}_tp_{i}" for i in range(u.shape[1])])
    sp_df = pd.DataFrame(sp, columns=[f"{nwp}_sp_{i}" for i in range(u.shape[1])])
    tcc_df = pd.DataFrame(tcc, columns=[f"{nwp}_tcc_{i}" for i in range(u.shape[1])])
    
    ws = np.sqrt(u ** 2 + v ** 2)
    ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(ws.shape[1])])
    nwp_df = pd.concat([u_df, v_df, ws_df, ghi_df, poai_df, t2m_df, tp_df, sp_df, tcc_df], axis=1)
    x_df = pd.concat([x_df, nwp_df], axis=1)
    return x_df