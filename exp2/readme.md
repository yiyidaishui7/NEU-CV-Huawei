# 实验2-1
## 实验要求
本地跑通deeplabv3

## 运行步骤如下：
1. 下载SBD数据集及VOC2012数据集，放到deeplabv3/src/data下

2. 实现语义数据集。
   运行deeplabv3/src/data/get_dataset_lst.py
   会得到三个txt：voc_train_lst.txt voc_val_lst.txt vocaug_train_lst.txt

3. 将数据集转换为MindRecord。运行build_seg_data.py（需配置参数
   
4. 模型训练。下载预训练模型resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt，放到deeplabv3/model下；运行train.py（需配置参数
   
5. 模型评估。运行eval.py（需配置参数
