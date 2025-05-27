
# WindSolarPowerForecast 新能源功率预测系统

## 项目简介
本仓库是一个新能源（风能/太阳能）功率预测的开源项目，基于数值天气预报（NWP）数据和机器学习模型，实现对风电场和光伏电站的功率预测。支持多场站、多模型训练及结果输出，适用于短期功率预测场景(新能源功率预测：http://competition.sais.com.cn/competitionDetail/532315/competitionData)。


## 目录结构
```
WindSolarPowerForecast/
├─ data/                # 原始数据目录（NWP数据、实际功率数据）
│  ├─ TRAIN/            # 训练数据集
│  └─ TEST/             # 测试数据集
├─ code/                # 代码核心目录
│  ├─ main/             # 主程序
│  │  ├─ main.py        # 程序入口
│  │  └─ utils.py       # 工具函数
│  └─ common/           # 公共模块
│     └─ utils.py       # 通用工具（数据处理、模型定义等）
├─ cache/               # 缓存数据目录（自动生成）
├─ output/              # 结果输出目录
│  ├─ models/           # 训练好的模型
│  └─ results/          # 预测结果文件
├─ requirements.txt     # 依赖清单
└─ README.md            # 项目说明
```


## 环境配置
### 依赖安装
```bash
pip install -r requirements.txt
```

### 关键依赖说明
- **数据处理**：pandas, xarray, scikit-learn
- **模型训练**：scikit-learn, xgboost（可扩展其他模型）
- **日志记录**：loguru
- **系统工具**：numpy, scipy, tqdm


## 快速开始
### 1. 数据准备
- **NWP数据**：放置在`data/TRAIN/nwp_data_train/`（训练）和`data/TEST/nwp_data_{test}/`（测试），按场站ID分文件夹存储。
- **实际功率数据**：命名为`{farm_id}_normalization_train.csv`，放置在`data/TRAIN/fact_path/`。

### 2. 配置参数
- **数据路径**：在`code/main/utils.py`中修改`base_path`为实际数据根目录。
- **模型参数**：在`train`函数中指定`model_name`（如"xgb"）、`power_type`（"ws"为风能，"pv"为太阳能）。

### 3. 运行程序
```bash
python code/main/main.py
```

- **首次运行**：会自动解析NC文件并生成CSV缓存到`cache/`目录。
- **训练结果**：模型保存至`output/models/{farm_id}/`，预测结果保存至`output/results/output_{train/test}/`。


## 功能说明
### 核心模块
1. **数据处理**：
   - `process_nwp_data`：解析NWP数据（支持多文件合并）。
   - `apply_feature_engineering`：特征工程（如时间特征、气象参数组合）。
   - `data_preprocess`：数据标准化、缺失值处理。

2. **模型训练**：
   - 支持XGBoost等机器学习模型，可通过`get_model`函数扩展其他模型。
   - 自动处理多场站训练，支持风能/太阳能场景切换。

3. **预测推理**：
   - 支持训练集和测试集预测，结果自动进行后处理（如负值截断、插值）。
   - 输出结果为CSV文件，按场站ID命名，包含时间序列预测值。


## 结果说明
- **输出文件结构**：
  ```
  output/
  ├─ models/
  │  └─ {farm_id}/        # 场站ID对应的模型文件
  └─ results/
     ├─ output_train/     # 训练集预测结果
     │  └─ output{farm_id}.csv  # 预测功率（时间索引，列名：power）
     └─ output_test/      # 测试集预测结果
        └─ output{farm_id}.csv
  ```
- **结果指标**：预测值范围为[0, 1]（归一化后），实际值需结合数据预处理中的归一化参数还原。


## 扩展与贡献
### 模型扩展
1. 在`common/utils.py`中添加新模型类，实现`fit`和`predict`接口。
2. 修改`get_model`函数，支持新模型的调用。

### 特征工程扩展
- 在`apply_feature_engineering`函数中添加新特征计算逻辑，通过`version`参数区分不同特征版本。