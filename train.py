import os
import codecs
import pickle
import itertools
from collections import OrderedDict

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner
from data_utils import create_input, BatchManager
import sentencepiece as spm

flags = tf.app.flags
flags.DEFINE_boolean("clean",       False,      "clean train folder")
flags.DEFINE_boolean("train",       False,      "Wither train the model")

# configurations for the model
flags.DEFINE_integer("batch_size",  2,         "batch size")
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    200,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iob",      "tagging schema iobes or iob")

# configurations for training
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.5,        "Dropout rate")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")

flags.DEFINE_integer("max_seq_len", 128,        "max sequence length for bert")
flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("train_file",   os.path.join("data", "example.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "example.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "example.test"),   "Path for test data")

flags.DEFINE_string("xlnet_config", default="chinese_xlnet_base_L-12_H-768_A-12/xlnet_config.json", help="xlnet config file dir")
flags.DEFINE_string("init_checkpoint", default="chinese_xlnet_base_L-12_H-768_A-12/xlnet_model.ckpt", help="xlnet model init checkpoint")
flags.DEFINE_string("spm", default="chinese_xlnet_base_L-12_H-768_A-12/spiece.model", help="spiece model file")
flags.DEFINE_string("entry", default="train", help="operation")
flags.DEFINE_float("dropatt", default=0.1, help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1, help="Clamp length.")
# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_bool("use_bfloat16", default=False, help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")


FLAGS = tf.app.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


# config for the model
def config_model(tag_to_id):
    config = OrderedDict()
    config["num_tags"] = len(tag_to_id)
    config["lstm_dim"] = FLAGS.lstm_dim
    config["batch_size"] = FLAGS.batch_size
    config['max_seq_len'] = FLAGS.max_seq_len

    config["clip"] = FLAGS.clip
    config["dropout_keep"] = 1.0 - FLAGS.dropout
    config["optimizer"] = FLAGS.optimizer
    config["lr"] = FLAGS.lr
    config["tag_schema"] = FLAGS.tag_schema
    config["zeros"] = FLAGS.zeros
    config["lower"] = FLAGS.lower
    return config

def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1

def train():
    # load data sets
    # 句子集合 = [[句子1],[句子2],[句子3]]，句子1 = [我 O，在 O，。。。]
    #<class 'list'>: [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']]
    # train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
    # dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
    # test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
    from xlnet_base.xlnet_data_utils import XLNetDataUtils
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load('./chinese_xlnet_base_L-12_H-768_A-12/spiece.model')

    train_data = XLNetDataUtils(sp_model, batch_size=FLAGS.batch_size,entry="train")
    dev_data = XLNetDataUtils(sp_model, batch_size=FLAGS.batch_size,entry="dev")
    test_data = XLNetDataUtils(sp_model, batch_size=FLAGS.batch_size, entry="test")
    dev_batch = dev_data.iteration()
    def datapadding(data):
        alldatalist = []

        datalist = data.data
        max_length = 64
        for i in range(len(datalist)):
            tmpdatalist = []
            token = datalist[i][0]
            segmentid = datalist[i][1]
            inputid = datalist[i][2]
            inputmask = datalist[i][3]
            labellist = datalist[i][4]
            #token label
            if len(labellist)<max_length:
                for i in range(max_length-len(token)):
                    labellist.append(0)
            elif len(labellist)>max_length:
                tmplabellist = []
                for i in range(max_length):
                    tmplabellist.append(labellist[i])
                labellist = tmplabellist
            #segmentid inputid inputmask
            if len(segmentid)<max_length:
                for i in range(max_length-len(segmentid)):
                    segmentid.append(0)
                    inputid.append(0)
                    inputmask.append(0)
            elif len(segmentid)>max_length:
                tmpsegmentid = []
                tmpinputid = []
                tmpinputmask = []
                for i in range(max_length):
                    tmpsegmentid.append(segmentid[i])
                    tmpinputid.append(inputid[i])
                    tmpinputmask.append(inputmask[i])
                segmentid = tmpsegmentid
                inputid = tmpinputid
                inputmask = tmpinputmask
            tmpdatalist.append(token)
            tmpdatalist.append(segmentid)
            tmpdatalist.append(inputid)
            tmpdatalist.append(inputmask)
            tmpdatalist.append(labellist)
            alldatalist.append(tmpdatalist)
        return alldatalist
    ftraindata = datapadding(train_data)

    fdevdata = datapadding(dev_data)
    ftestdata = datapadding(test_data)
    print(len(ftraindata))
    print(len(fdevdata))
    print(len(ftestdata))
    # traindata = {
    #     "batch_size": train_data.batch_size,
    #     "input_size": train_data.input_size,
    #     "vocab": train_data.vocab,
    #     "tag_map": train_data.tag_map,
    # }
    # devdata = {
    #     "batch_size": dev_data.batch_size,
    #     "input_size": dev_data.input_size,
    #     "vocab": dev_data.vocab,
    #     "tag_map": dev_data.tag_map,
    # }
    # testdata = {
    #     "batch_size": test_data.batch_size,
    #     "input_size": test_data.input_size,
    #     "vocab": test_data.vocab,
    #     "tag_map": test_data.tag_map,
    # }
    # if not os.path.exists("./model/train_data_map.pkl"):
    #     f = open("./model/train_data_map.pkl", "wb")
    #     pickle.dump(traindata, f)
    #     f.close()
    # if not os.path.exists("./model/dev_data_map.pkl"):
    #     f = open("./model/dev_data_map.pkl", "wb")
    #     pickle.dump(devdata, f)
    #     f.close()
    # if not os.path.exists("./model/test_data_map.pkl"):
    #     f = open("./model/test_data_map.pkl", "wb")
    #     pickle.dump(testdata, f)
    #     f.close()

    # Use selected tagging scheme (IOB / IOBES)
    #update_tag_scheme(train_sentences, FLAGS.tag_schema)
    #update_tag_scheme(test_sentences, FLAGS.tag_schema)


    # create maps if not exist
    if not os.path.isfile(FLAGS.map_file):
        # Create a dictionary and a mapping for tags
        '''
         _t:{'O': 869087, 'B-LOC': 16571, 'I-LOC': 22531, 'B-PER': 8144, 'I-PER': 15881, 'B-ORG': 9277, 'I-ORG': 37689, '[SEP]': 8, '[CLS]': 10}
         id_to_tag:{0: 'O', 1: 'I-ORG', 2: 'I-LOC', 3: 'B-LOC', 4: 'I-PER', 5: 'B-ORG', 6: 'B-PER', 7: '[CLS]', 8: '[SEP]'}
         tag_to_id:{'O': 0, 'I-ORG': 1, 'I-LOC': 2, 'B-LOC': 3, 'I-PER': 4, 'B-ORG': 5, 'B-PER': 6, '[CLS]': 7, '[SEP]': 8}
        '''

        tag_to_id = train_data.tag_map
        id_to_tag = {v: k for k, v in tag_to_id.items()}
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    '''
    [['在', '这', '里', '恕', '弟', '不', '恭', '之', '罪', '，', '敢', '在', '尊', '前', '一', '诤', '：', '前', '人', '论',
    '书', '，', '每', '曰', '“', '字', '字', '有', '来', '历', '，', '笔', '笔', '有', '出', '处', '”', '，', '细', '读', '公', 
    '字', '，', '何', '尝', '跳', '出', '前', '人', '藩', '篱', '，', '自', '隶', '变', '而', '后', '，', '直', '至', '明', '季',
    '，', '兄', '有', '何', '新', '出', '？'], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [101, 1762, 6821, 7027, 2609, 2475, 679, 2621, 722,
    5389, 8024, 3140, 1762, 2203, 1184, 671, 6420, 8038, 1184, 782, 6389, 741, 8024, 3680, 3288, 100, 2099, 2099, 3300,
    3341, 1325, 8024, 5011, 5011, 3300, 1139, 1905, 100, 8024, 5301, 6438, 1062, 2099, 8024, 862, 2214, 6663, 1139, 
    1184, 782, 5974, 5075, 8024, 5632, 7405, 1359, 5445, 1400, 8024, 4684, 5635, 3209, 2108, 8024, 1040, 3300, 862, 
    3173, 1139, 8043, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    '''

    # train_data = prepare_dataset(
    #     train_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    # )
    # dev_data = prepare_dataset(
    #     dev_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    # )
    # test_data = prepare_dataset(
    #     test_sentences, FLAGS.max_seq_len, tag_to_id, FLAGS.lower
    # )

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data.data), len(dev_data.data), len(test_data.data)))

    train_manager = BatchManager(ftraindata, FLAGS.batch_size)
    dev_manager = BatchManager(fdevdata, FLAGS.batch_size)
    test_manager = BatchManager(ftestdata, FLAGS.batch_size)
    # make path for store log and model if not exist
    make_path(FLAGS)
    if os.path.isfile(FLAGS.config_file):
        config = load_config(FLAGS.config_file)
    else:
        config = config_model(tag_to_id)
        save_config(config, FLAGS.config_file)
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    with tf.Session(config=tf_config) as sess:
        model = create_model(sess, Model, FLAGS.ckpt_path, config, logger)
        logger.info("start training")
        loss = []
        for i in range(100):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)

                loss.append(batch_loss)
                if step % FLAGS.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS.ckpt_path, logger, global_steps=step)
            evaluate(sess, model, "test", test_manager, id_to_tag, logger)

def main(_):
    FLAGS.train = True
    FLAGS.clean = True
    clean(FLAGS)
    train()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    tf.app.run(main)