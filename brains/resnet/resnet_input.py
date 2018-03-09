import tensorflow as tf

#宽度
WIDTH = 8
#高度
HEIGHT = 8
#单层灰度图
DEPTH = 1
#默认label集
num_classes = 10
#默认label byte长度
label_bytes = 1
#label offset byte长度
label_offset = 0

#乱序提取部分样本
def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 3000
    elif subset == 'validation':
      return 600
    elif subset == 'eval':
      return 600
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

def preprocess(image):
  """预处理一个图像，按照[height, width, depth]组织""" 
  image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
  image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
  image = tf.image.random_flip_left_right(image)
  return image

def parser(serialized_example):
  """逐个转换example"""
  features = tf.parse_single_example(
    serialized_example,
    features={
       'image': tf.FixedLenFeature([], tf.string),
       'label': tf.FixedLenFeature([], tf.int64),
    })
  image = tf.decode_raw(features['image'], tf.uint8)
  image.set_shape([DEPTH * HEIGHT * WIDTH])
  image = tf.cast(tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),tf.float32)
  label = tf.cast(features['label'], tf.int32)
  image = preprocess(image)
  return image,label


# data_path -> records文件路径
# batch_size -> 批尺寸，即学习一次使用的数据集中数据个数
# num_classes -> label大小，如分为10类 num_classes = 10
# mode -> train or eval,设置样本是训练集还是测试集
def build_input(data_path, batch_size, num_classes, mode):
  #读取dataset
  dataset = tf.data.TFRecordDataset(data_path)
  dataset = dataset.map(parser)
  #处理train样本集
  if mode == 'train':
    #预计数量*0.4构建随机乱序batch
    min_queue_examples= int(num_examples_per_epoch(mode)*0.4)
    #确保整体容量能够生成较为优质的batch
    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
  #打包
  dataset = dataset.batch(batch_size)
  #构建迭代tesnor
  iterator = dataset.make_one_shot_iterator()
  #获取图片tensor和标注tensor
  image_batch, label_batch = iterator.get_next()
  #返回tensor
  return image_batch, label_batch

if __name__ == '__main__':
  train = 'D:/Workspace/train/train.tfrecords'
  build_input(train,10,100,'train')
