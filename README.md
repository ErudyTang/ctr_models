# 一些基本ctr预估模型的tensorflow和pytorch实现
## 环境要求
* python==3.8.5  
* tensorflow==2.4.1  
* pytorch==1.10.1  
* 如果要运行tensorflow Estimator模型demo，tensorflow的版本为1.14.0  

## 配置修改
* 修改conf/xx_conf.json中的「home_dir」符合该项目实际所在路径

## Demo运行
* tensorflow-keras：cd tensorflow && python keras_train.py dnn ../conf/tf_conf.json
* tensorflow-estimator：cd tensorflow && python estimator_train.py dnn ../conf/tf_conf.json
* pytorch：cd pytorch && python train.py dnn ../conf/th_conf.json
