## 基于TensorFlow的Embedding神经网络Rossmann销售预测

## 软件要求
* Python 3.x
* Keras 1.2.2
* TensorFlow
* matplotlib

## 数据说明
项目目录data中包括了项目需要的全部数据，其中store_states.csv来自https://www.kaggle.com/c/rossmann-store-sales/discussion/17048，其他数据来自Kaggle项目。

## 项目运行
本项目运行可以采用两种方式：
 1. Jupyter notebooks
 2. Python scripts

### Jupyter notebook运行
运行rossmann-report.ipynb, 按步骤执行。

### Jupyter notebook运行
安装要求的软件之后，本项目执行需要以下几个步骤：
 1. 准备数据
python prepare_data.py

 2. 训练并执行模型
python model.py
