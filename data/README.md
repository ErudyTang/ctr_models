# 对原始训练数据进行预处理
## 数据集说明
数据集：Criteo，全量数据下载地址：https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/10082655/dac.tar.gz
* 取train.txt的前1000000行作为训练集，即raw/train_1m.txt
* 取train.txt的最后100000行作为测试集，即raw/test_100k.txt
* 取train.txt的倒数100000~200000行作为验证集，即raw/val_100k.txt

## 预处理方式
* 对于int类型特征，首先进行归一化$v=\log(v-mn+1)/\log(mx-mn+1)$，然后离散化到[0, 1000]，其中mn、mx分别表示该特征值在训练集上可以取到的最小、最大值
* 对于cat类型特征，去掉低频类别（在训练集上出现的次数小于某个阈值）

因为pytorch不支持string类型的tensor，为了方便，在预处理过程中将string映射为id。  
针对tensorflow训练流程：  
```
mkdir train val
python preprocess.py raw/train_1m.txt sta.join fit  
python preprocess.py raw/train_1m.txt sta.join transfrom --outfilename train/train_1m.txt  
python preprocess.py raw/val_100k.txt sta.join transform --outfilename val/val_100k.txt  
```
针对pytorch训练流程：  
```
mkdir train_th val_th
python preprocess.py raw/train_1m.txt sta.join fit  
python preprocess.py raw/train_1m.txt sta.join transfrom --outfilename train_th/train_1m.txt --str2idx True  
python preprocess.py raw/val_100k.txt sta.join transform --outfilename val_th/val_100k.txt --str2idx True  
```

上述训练集、验证集的目录名需要与../conf/xx_conf.json中的配置匹配。  