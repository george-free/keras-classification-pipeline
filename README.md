# cv_classification_pipeline
This is a repo for putting classification model and training code

具体描述请参考以下文档：

https://docs.qq.com/doc/DTEZMRHpmalFwYWdk


网络的一切参数配置和设置都在＂config.py＂里面，主要包括设置以下三种类型的参数：

１）数据来源和数据类型．

２）网络名称和网络的参数．

３）中间结果存放路径．


[训练步骤]

1. 安装虚拟环境

$ virtualenv -p python3 venv

$ source venv/bin/activate

2. 安装必要库环境

$ pip install -r requirements.txt

3. 拷贝"config.py.example"到"config.py"，修改"config.py"里面的相关配置，设置网络．

$ cp config.py.example config.py

4. 训练

$ python3 train.py

5. 在log文件中查看，训练的train/valid结果图．

[数据集] 
  
  程序里面的数据处理部分已经做好，目前支持两种数据源，

1. CIFAR

下载地址：

https://cv-internal.s3.cn-north-1.amazonaws.com.cn/open-datasets/CIFAR-DATASET/CIFAR-10/batches.meta

２．自定义的图像数据

图像按照类别放在各自类别的文件夹下面，如下面这个目录结构．

train \
       dog \
             train_dog_1.png
             train_dog_2.png
             ...
       cat \
             train_cat_1.png
             train_cat_2.png
             
valid \
       dog \ 
             valid_dog_1.png
             valid_dog_2.png
             ...
       cat \
             valid_cat_1.png
             valid_cat_2.png

test \
      dog \ 
             test_dog_1.png
             test_dog_2.png
             ...
      cat \
             test_cat_1.png
             test_cat_2.png

[模型转化]

Keras 模型能够直接转化未tensorflow 模型．

通过执行　tools/freeze_model.py，

$ cd tools

$ python3  freeze_model.py  -i  /path/your_model.h5  -o  /path_to_save_pb -a additional_model_name
# 有些网络用了自定义层, 例如resnet50-stn, 所以需要加上 -a 来指定.
# e.g.
# python3  freeze_model.py  -i  /path/your_model.h5  -o  /path_to_save_pb -a BilinearInterpolation


[自定义分类模型]

1. 继承父亲类＂ClassificationModel＂, 并实现"build_network"方法．

2. 在 "model_factory.py"里面添加自定义的这个类．

3. 在 ＂config.py＂里面配置这个累进行训练．
