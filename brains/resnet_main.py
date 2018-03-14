#save 模型示例
#http://blog.csdn.net/thriving_fcl/article/details/71423039

import time
import six
import sys

import numpy as np
import resnet_model
import tensorflow as tf
import resnet_input
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
#设置参数
FLAGS = tf.app.flags.FLAGS
#
tf.app.flags.DEFINE_integer('batch_size', 128, '一批数据的容量')
tf.app.flags.DEFINE_integer('num_classes', 11, 'label类型长度')
tf.app.flags.DEFINE_string('mode', 'train', 'train训练，eval测试')
tf.app.flags.DEFINE_string('train_data_path', 'workspace/', 'train.records地址')
tf.app.flags.DEFINE_string('eval_data_path', 'workspace/', 'eval.records地址')
tf.app.flags.DEFINE_integer('width', 10, '图片宽度')
tf.app.flags.DEFINE_integer('height', 10, '图片高度')
tf.app.flags.DEFINE_integer('depth', 1, '通道')
tf.app.flags.DEFINE_string('train_dir', 'workspace/tfrecords/All13x10/tmp/', '训练输出')
tf.app.flags.DEFINE_string('eval_dir', 'workspace/tfrecords/All13x10/tmp/', '验证输出')
tf.app.flags.DEFINE_integer('eval_batch_count', 100, '验证机一批样本数量')
tf.app.flags.DEFINE_bool('eval_once', False, 'Whether evaluate the model only once.')
tf.app.flags.DEFINE_string('log_root', 'workspace/tmp/', 'Directory to keep the checkpoints. Should be a ''parent directory of FLAGS.train_dir/eval_dir.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'Number of gpus used for training. (0 or 1)')
#训练模式
def train(hps):
  """Training loop."""
  #构建images和labels训练input
  input = resnet_input.ResnetInput(data_dir=FLAGS.train_data_path, image_width = FLAGS.width, image_height = FLAGS.height, image_depth=FLAGS.depth)
  #make
  images_batch,label_batch = input.make_batch(FLAGS.batch_size)
  #初始化resnet参数
  model = resnet_model.ResNet(hps, images_batch, label_batch, FLAGS.width, FLAGS.height, FLAGS.depth ,FLAGS.mode)
  #构建tensorflow graph
  model.build_graph()
  #分析训练参数
  param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(tf.get_default_graph(), tfprof_options=tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  #打印训练参数
  sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
  #argmax 返回最大值的索引号 例如 [[1,3,4,5,6]] - > [4] 或 [[1,3,4], [2,4,1]] -> [2,1]
  #truth = tf.argmax(model.labels, axis=0)
  truth = tf.cast(model.labels,tf.int64)
  #输出值
  predictions = tf.argmax(model.predictions, axis=1)
  #精度验证
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
  #结论，提供给tensorboard可视化
  summary_hook = tf.train.SummarySaverHook(save_steps=100, output_dir=FLAGS.train_dir, summary_op=tf.summary.merge([
                                           model.summaries, tf.summary.scalar('Precision', precision)]))
  #结论，打印过程
  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step, 'loss': model.cost, 'precision': precision,'predictions':predictions,'truth':truth}, 
      every_n_iter=100)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.1

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      train_step = run_values.results
      if train_step < 100000:
        self._lrn_rate = 0.1
      elif train_step < 200000:
        self._lrn_rate = 0.01
      elif train_step < 300000:
        self._lrn_rate = 0.001
      else:
        self._lrn_rate = 0.0001

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=FLAGS.log_root,
      hooks=[logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=1000,config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)


def evaluate(hps):
  """Eval loop."""
  images, labels = resnet_input.build_input(
      FLAGS.eval_data_path, hps.batch_size, FLAGS.num_classes, FLAGS.mode)
  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  model.build_graph()
  saver = tf.train.Saver()
  summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  tf.train.start_queue_runners(sess)
  best_precision = 0.0

  while True:
    try:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    except tf.errors.OutOfRangeError as e:
      tf.logging.error('Cannot restore checkpoint: %s', e)
      continue
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to eval yet at %s', FLAGS.log_root)
      continue
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    total_prediction, correct_prediction = 0, 0
    for _ in six.moves.range(FLAGS.eval_batch_count):
      (summaries, loss, predictions, truth, train_step) = sess.run(
          [model.summaries, model.cost, model.predictions,
           model.labels, model.global_step])

      truth = np.argmax(truth, axis=1)
      predictions = np.argmax(predictions, axis=1)
      correct_prediction += np.sum(truth == predictions)
      total_prediction += predictions.shape[0]

    precision = 1.0 * correct_prediction / total_prediction
    best_precision = max(precision, best_precision)

    precision_summ = tf.Summary()
    precision_summ.value.add(
        tag='Precision', simple_value=precision)
    summary_writer.add_summary(precision_summ, train_step)
    best_precision_summ = tf.Summary()
    best_precision_summ.value.add(
        tag='Best Precision', simple_value=best_precision)
    summary_writer.add_summary(best_precision_summ, train_step)
    summary_writer.add_summary(summaries, train_step)
    tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                    (loss, precision, best_precision))
    summary_writer.flush()

    if FLAGS.eval_once:
      break

    time.sleep(60)


def main(_):
  #dvice判断
  if FLAGS.num_gpus == 0:
    dev = '/cpu:0'
  elif FLAGS.num_gpus == 1:
    dev = '/gpu:0'
  else:
    raise ValueError('Only support 0 or 1 gpu.')

  #hparams设置
  hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                             num_classes=FLAGS.num_classes,
                             min_lrn_rate=0.0001,
                             lrn_rate=0.1,
                             num_residual_units=5,
                             use_bottleneck=False,
                             weight_decay_rate=0.0002,
                             relu_leakiness=0.1,
                             optimizer='sgd')
  #训练模型
  with tf.device(dev):
    if FLAGS.mode == 'train':
      train(hps)
    elif FLAGS.mode == 'eval':
      evaluate(hps)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #解析flag,执行module里的main
  tf.app.run()
