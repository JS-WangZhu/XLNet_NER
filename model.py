# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.layers.python.layers import initializers
from absl import flags
import rnncell as rnn
from utils import bio_to_json
from bert import modeling
from xlnet_base import modeling, xlnet, model_utils
FLAGS = flags.FLAGS
class Model(object):
    def __init__(self, config):

        self.config = config
        self.lr = config["lr"]
        self.lstm_dim = 200
        self.num_tags = config["num_tags"]
        self.checkpoint_dir = "./model/"
        self.checkpoint_path = "./model/ner.ckpt"

        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        # # add placeholders for the model
        # self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
        # self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_mask")
        # self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")
        # self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")
        # # dropout keep prob
        # self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.float32,
            shape=[None, None],
            name="xlnet_input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_segment_ids"
        )
        self.targets = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name="xlnet_targets"
        )

        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="xlnet_dropout"
        )


        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)
        self.batch_size = tf.shape(self.input_ids)[0]
        self.num_steps = tf.shape(self.input_ids)[-1]

        print(self.num_steps)
        print(self.batch_size)
        # embeddings for chinese character and segmentation representation
        #embedding = self.bert_embedding()
        embedding = self.xlnet_layer()
        print(embedding.shape)
        # apply dropout before feed to lstm layer
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)

        # bi-directional lstm layer
        lstm_outputs = self.biLSTM_layer(lstm_inputs, self.lstm_dim, self.lengths)

        # logits for tags
        self.logits = self.project_layer(lstm_outputs)
        #self.logits = self.attention(lstm_outputs,100)
        #atto = self.attention(lstm_outputs,100)

        # print(atto.shape)
        # loss of the model
        self.loss = self.loss_layer(self.logits, self.lengths)

        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    print("restore model")
                    self.saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                    tvars = tf.trainable_variables()
                    (assignment_map, initialized_variable_names) = model_utils.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint)
                    tf.train.init_from_checkpoint(FLAGS.init_checkpoint, assignment_map)
                    for var in tvars:
                        init_string = ""
                        if var.name in initialized_variable_names:
                            init_string = ", *INIT_FROM_CKPT*"
                        print("  name = %s, shape = %s%s", var.name, var.shape,init_string)
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            else:
                raise KeyError

            grads = tf.gradients(self.loss, tvars)
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            self.train_op = self.opt.apply_gradients(
                zip(grads, tvars), global_step=self.global_step)
            #capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
            #                     for g, v in grads_vars if g is not None]
            #self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step, )

        # saver of the model
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    # def bert_embedding(self):
    #     # load bert embedding
    #     bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")  # 配置文件地址。
    #     model = modeling.BertModel(
    #         config=bert_config,
    #         is_training=True,
    #         input_ids=self.input_ids,
    #         input_mask=self.input_mask,
    #         token_type_ids=self.segment_ids,
    #         use_one_hot_embeddings=False)
    #     embedding = model.get_sequence_output()
    #     return embedding

    def xlnet_layer(self):
        from xlnet_base import xlnet
        xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.xlnet_config)
        run_config = xlnet.create_run_config(True, True, FLAGS)
        xlnet_model = xlnet.XLNetModel(
            xlnet_config=xlnet_config,
            run_config=run_config,
            input_ids=self.input_ids,
            seg_ids=self.segment_ids,
            input_mask=self.input_mask)
        embedding = xlnet_model.get_sequence_output()
        embedding = tf.nn.dropout(
            embedding, self.dropout
        )
        return embedding


    def biLSTM_layer(self, lstm_inputs, lstm_dim, lengths, name=None):
        """
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM" if not name else name):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        lstm_dim,
                        use_peepholes=True,
                        initializer=self.initializer,
                        state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                tf.contrib.rnn.AttentionCellWrapper(lstm_cell["forward"],attn_length=50),
                tf.contrib.rnn.AttentionCellWrapper(lstm_cell["backward"],attn_length=50),
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs, name=None):
        """
        hidden layer between lstm layer and logits
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"  if not name else name):
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, W, b))

            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, W, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def attention(self, inputs, attention_size, time_major=False, return_alphas=False):
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = tf.concat(inputs, 2)
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

        hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer 200

        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size(128,128,400)*(400,100)=(128,128,100)
            v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape (128,128)

        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape (128,128)
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape (128,400)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

        attentionout = tf.reshape(output, shape=[self.batch_size, -1])
        with tf.variable_scope("project11"):
            print(type(self.num_steps))
            with tf.variable_scope("hidden"):
                W = tf.get_variable("W", shape=[self.lstm_dim * 2, self.num_steps], dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_steps], dtype=tf.float32, initializer=tf.zeros_initializer())
                hidden1 = tf.tanh(tf.nn.xw_plus_b(attentionout, W, b))

        with tf.variable_scope("project12"):
            hidden2 = tf.reshape(hidden1, shape=[self.num_steps * self.num_steps, -1])
            with tf.variable_scope("hidden1"):
                W = tf.get_variable("W", shape=[1, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                hidden2 = tf.tanh(tf.nn.xw_plus_b(hidden2, W, b))

            with tf.variable_scope("logits1"):
                W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                output = tf.nn.xw_plus_b(hidden2, W, b)
                output = tf.reshape(output, shape=[-1, 128, self.num_tags])
        return output

    # def project_att_layer(self, attentionout, name=None):
    #     """
    #     隐藏层 间于Attention层和输出层
    #     """
    #
    #     attentionout = tf.reshape(attentionout, shape=[self.batch_size, -1])
    #     with tf.variable_scope("project11" if not name else name):
    #         print(type(self.num_steps))
    #         with tf.variable_scope("hidden"):
    #             W = tf.get_variable("W", shape=[self.lstm_dim * 2, self.num_steps], dtype=tf.float32, initializer=self.initializer)
    #
    #             b = tf.get_variable("b", shape=[self.num_steps], dtype=tf.float32,initializer=tf.zeros_initializer())
    #             hidden1 = tf.tanh(tf.nn.xw_plus_b(attentionout, W, b))
    #
    #     with tf.variable_scope("project12" if not name else name):
    #         hidden2 = tf.reshape(hidden1, shape=[self.num_steps*self.num_steps, -1])
    #         with tf.variable_scope("hidden1"):
    #             W = tf.get_variable("W", shape=[1, self.lstm_dim],
    #                                 dtype=tf.float32, initializer=self.initializer)
    #
    #             b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #             hidden2 = tf.tanh(tf.nn.xw_plus_b(hidden2, W, b))
    #
    #         with tf.variable_scope("logits1"):
    #             W = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
    #                                 dtype=tf.float32, initializer=self.initializer)
    #
    #             b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
    #                                 initializer=tf.zeros_initializer())
    #
    #             output = tf.nn.xw_plus_b(hidden2, W, b)
    #             output = tf.reshape(output, shape=[-1, self.num_steps, self.num_tags])
    #     return output
    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"  if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            print(type(self.num_steps))
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        :param is_train: Flag, True for train batch
        :param batch: list train/evaluate data 
        :return: structured data to feed
        """
        _, segment_ids, chars, mask, tags = batch
        feed_dict = {
            self.input_ids: np.asarray(chars),
            self.input_mask: np.asarray(mask),
            self.segment_ids: np.asarray(segment_ids),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        :param sess: session to run the batch
        :param is_train: a flag indicate if it is a train batch
        :param batch: a dict containing batch data
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run(
                [self.global_step, self.loss, self.train_op],
                feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags +[0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths

    def evaluate(self, sess, data_manager, id_to_tag):
        """
        :param sess: session  to run the model 
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = self.trans.eval()
        for batch in data_manager.iter_batch():
            strings = batch[0]
            labels = batch[-1]
            lengths, scores = self.run_step(sess, False, batch)
            batch_paths = self.decode(scores, lengths, trans)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                gold = [id_to_tag[int(x)] for x in labels[i][1:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][1:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results

    def evaluate_line(self, sess, inputs, id_to_tag):
        trans = self.trans.eval(sess)
        lengths, scores = self.run_step(sess, False, inputs)
        batch_paths = self.decode(scores, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return bio_to_json(inputs[0], tags[1:-1])