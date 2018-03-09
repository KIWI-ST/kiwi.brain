import tensorflow as tf

#宽度
width = 32
#高度
height = 32
#单层灰度图
depth = 1
#默认label集
num_classes = 10
#默认label byte长度
label_bytes = 1
#label offset byte长度
label_offset = 0

# data_path -> records文件路径
# batch_size -> 批尺寸，即学习一次使用的数据集中数据个数
# num_classes -> label大小，如分为10类 num_classes = 10
# mode -> train or eval,设置样本是训练集还是测试集
def build_input(data_path, batch_size, num_classes, mode):
  #图片byet长度
  image_bytes = width * height * depth
  #record的总长度
  record_bytes = label_bytes + label_offset + image_bytes
  #数据读取
  data_files = tf.gfile.Glob(data_path)
  #根据文件名生成一个队列
  file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  #读取样本
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(file_queue)
  #样本处理
  record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
  #样本标注
  label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
  #转换[width*height*depth] -> [depth,width,height]
  depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),[depth, width, height])
  #转换 [depth, height, width] -> [height, width, depth].
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  #处理train样本集
  if mode == 'train':
    image = tf.image.resize_image_with_crop_or_pad(image, width + 4, height + 4)
    image = tf.random_crop(image, [width, height, 3])
    image = tf.image.random_flip_left_right(image)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)
    #构建训练样本
    example_queue = tf.RandomShuffleQueue(
        capacity=16 * batch_size,
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[width, height, depth], [1]])
    num_threads = 16
  else:
    image = tf.image.resize_image_with_crop_or_pad(image, width, height)
    image = tf.image.per_image_standardization(image)
    #构建测试样本
    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[width, height, depth], [1]])
    num_threads = 1
  #图操作
  example_enqueue_op = example_queue.enqueue([image, label])
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(example_queue, [example_enqueue_op] * num_threads))
  #读取image和labels
  images, labels = example_queue.dequeue_many(batch_size)
  labels = tf.reshape(labels, [batch_size, 1])
  indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
  labels = tf.sparse_to_dense(tf.concat(values=[indices, labels], axis=1),[batch_size, num_classes], 1.0, 0.0)
  #验证数据集
  assert len(images.get_shape()) == 4
  assert images.get_shape()[0] == batch_size
  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 2
  assert labels.get_shape()[0] == batch_size
  assert labels.get_shape()[1] == num_classes
  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, labels
