# FinTech_DefaultRisk

## 文件说明

* /data

  * application_{train | test}.csv

    原始数据

  * processed_{train | test}.csv

    经过基本预处理、特征工程后的数据

* /result

  * fi.csv

    特征重要性

  * log_reg_lr.csv

  * random_forest_baseline.csv

* vis.ipynb

  读取数据，并进行可视化探索。没有修改数据。

* feature_engineering.ipynb

  读取application_{train | test}.csv文件，进行基本的预处理、特征工程，并保存为processed_{train | test}.csv文件。

* Baseline(LR).ipynb

  实现逻辑回归模型。

* Baseline(RF).ipynb

  实现随机森林模型。

* lightGRMTest.ipynb

  实现LGRM模型。