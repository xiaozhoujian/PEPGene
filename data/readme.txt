rawData: 原始数据
aeData: pep自编码器的训练集和测试集
mainData: 主训练的训练集和测试集
forcast: (应用阶段)生成结果

cleanData.py: 读取rawData，对三类数据分组1-k-k的同时：
1.随机选取一些pep放到aeData/eval.txt和aeData/train.txt中
2.将数据分配到mainData/eval/和mainData/train/中

mainData/encode.py：预训练完成后用PEP编码器对主训练的训练集和测试集进行编码