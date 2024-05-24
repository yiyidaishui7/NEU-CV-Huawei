# NEU&HuaweiCloud
## 实验目的
熟悉 modelarts开发平台，掌握本地连接modelarts方法，了解深度而学习训练及推理流程。

实验环境搭建

安装mindspore、opencv-python。使用pip命令安装。

使用python -c "import mindspore;mindspore.run_check()"验证是否安装成功。

登录OBS。

## 实验2.1

在本地及modelarts上跑通lenet网络训练及验证代码。

## 实验2.2

在本地跑通deeplabv3网络训练及评估代码。要求:提交训练过程及结果截图
参考教程:基于MindStudio搭建DeeplabV3网络实现图像语义分割任务.docx

1. 在GitLee下载开源代码。位于models-1.5/official/cv/deeplabv3下。

2. 下载数据集。包括SBD数据集和VOC数据集。将数据集文件放置到deeplabv3\src\data\

3. 实现语义数据集图像RGB图转换为灰度图、对数据集划分生成训练集，测试集、增强数据集，按固定格式写入txt文件。
运行deeplabv3\src\data\get_dataset_lst.py。

4. 查看生成的文件。

5. 将数据集转换为MindRecord，运行build_seg_data.py。

6. 训练模型。运行train.py。
配置参数
--data_file=D:\code\models-r1.5\official\cv \deeplabv3\src\data\preprocess\MindRecoder_train0 #修改自己的mindRecod 数据路径 

--device_target=CPU #使用芯片类型，可以选cpu,Ascend,gpu

--train_dir=./ckpt #存储模型文件路径

--train_epochs=20 #训练轮数

--batch_size=32 ##根据自己电脑配置进行修改，16，8等

--crop_size=513 #图片尺寸

--base_lr=0.015 #学习率

--lr_type=cos #

--min_scale=0.5

--max_scale=2.0

--ignore_label=255

--num_classes=21 #类别数

--model=deeplab_v3_s16 #模型类型

--ckpt_pre_trained=D:\code\models-r1.5\official\cv\deeplabv3\model\resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt #预训练模型路径

--save_steps=3 # 训练多少步保存一次模型，请注意修改

--keep_checkpoint_max=200 # 最多保存多少个模型文件


7. 模型评估。运行eval.py。
修改参数。

--data_root=.\src\data # 数据路径注意修改

--data_lst= D:\code\models-r1.5\official\cv\deeplabv3\src\data\voc_val_lst.txt #验证数据集路径

--batch_size=16 

--device_target=CPU

--crop_size=513

--ignore_label=255

--num_classes=21

--model=deeplab_v3_s8

--ckpt_path= D:\code\models-r1.5\official\cv\deeplabv3\ckpt\deeplab_v3_s16-1_12.ckpt


如果设备设置错误。第33行 Ascend修改为CPU 本代码已修改


## 选做

在 modelarts上跑通deeplabv3网络训练及评估代码。
要求:提交训练过程及结果截图。
提示:使用obs 上传代码及数据集，并使用mox.py 传送代码。

创建OBS桶，上传数据。
执行mox.py，将文件和数据集从桶中取出。

