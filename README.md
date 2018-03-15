# kiwi.brain
deep learning to resolve trace network generation / trajectory correction ...

## description ##
>使用resnet对样本进行训练，得到的结果可以在 kiwi.server中，对图像进行分类操作 <br />

## features ##
>1.train.records 构建了分类结果为11类的样本 <br />
>2.样本图片的尺寸为 10x10，通道为1 <br />
>3.样本尺寸尺寸可自定义修改<br />

## usage ###
>1.构建records样本,使用 src/utils/build_records.py 将样本图片和label构建train.tfrecords和eval.tfrecords
>2.使用src/brains/resnet_main.py 训练模型，注意tf.Flags修改相关参数，常修改的参数是样本图片的规格
