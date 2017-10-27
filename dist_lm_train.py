"""
Distributed LM training
"""
import os

import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval

tf.flags.DEFINE_string("role", "ps", "ps or worker")
tf.flags.DEFINE_string("ps_list", "", "ps list of hosts")
tf.flags.DEFINE_string("worker_list", "", "worker list of hosts")
tf.flags.DEFINE_integer("task_index", 0, "index of job")
tf.flags.DEFINE_string("logdir", "lm1b", "Logging directory.")
tf.flags.DEFINE_string("datadir", None, "Logging directory.")
tf.flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
tf.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.flags.DEFINE_integer("num_gpus", 8, "Number of GPUs used.")
tf.flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")


FLAGS = tf.flags.FLAGS
cluster_spec = {"ps": FLAGS.ps_list.split(','), "worker": FLAGS.worker_list.split(',') }

cluster = tf.train.ClusterSpec(cluster_spec)
server = tf.train.Server(cluster, job_name=FLAGS.role, task_index=FLAGS.task_index)
if FLAGS.role == "ps":
    server.join()
else:
    ps_device = '/job:ps/task:0'
    """
    Start either train or eval. Note hardcoded parts of path for training and eval data
    """
    hps = LM.get_default_hparams().parse(FLAGS.hpconfig)
    hps._set("num_gpus", FLAGS.num_gpus)
    print('*****HYPER PARAMETERS*****')
    print(hps)
    print('**************************')

    vocab = Vocabulary.from_file(os.path.join(FLAGS.datadir, "1b_word_vocab.txt"))

    if FLAGS.mode == "train":
        #hps.batch_size = 256
        dataset = Dataset(vocab, os.path.join(FLAGS.datadir,
                                              "training-monolingual.tokenized.shuffled/*"))
        run_train(dataset, hps, os.path.join(FLAGS.logdir, "train"), ps_device=ps_device)

