"""
Distributed LM training
"""
import os

import tensorflow as tf
from data_utils import Vocabulary, Dataset
from language_model import LM
from run_utils import run_train, run_eval

tf.flags.DEFINE_string("my_ip", "", "ip")
tf.flags.DEFINE_string("ps_list", "", "ps list of hosts")
tf.flags.DEFINE_string("worker_list", "", "worker list of hosts")
tf.flags.DEFINE_string("logdir", "lm1b", "Logging directory.")
tf.flags.DEFINE_string("datadir", None, "Logging directory.")
tf.flags.DEFINE_string("mode", "train", "Whether to run 'train' or 'eval' model.")
tf.flags.DEFINE_string("hpconfig", "", "Overrides default hyper-parameters.")
tf.flags.DEFINE_integer("num_gpus", 8, "Number of GPUs used.")
tf.flags.DEFINE_integer("eval_steps", 70, "Number of eval steps.")


role = ''
task_index = -1
FLAGS = tf.flags.FLAGS
cluster_spec = {"ps": FLAGS.ps_list.split(','), "worker": FLAGS.worker_list.split(',') }

# check which one we are
for psi, psname in enumerate(cluster_spec['ps']):
    if psname.startswith(FLAGS.my_ip):
        role = 'ps'
        task_index = psi
        break

if role == '':
    for workeri, workername in enumerate(cluster_spec['worker']):
        if workername.startswith(FLAGS.my_ip):
            role = 'worker'
            task_index = workeri
            break

if role == '':
    print("Your IP is %s" % FLAGS.my_ip)
    print("Cluster spec is %s"%str(cluster_spec))
    print("You're not in the cluster spec!  exiting!")
    exit(-1)
else:
    print("ROLE: %s" % role)
    print("INDEX: %s" % task_index)

cluster = tf.train.ClusterSpec(cluster_spec)
server = tf.train.Server(cluster, job_name=role, task_index=task_index)
if role == "ps":
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

