# blog: https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html
#   https://guillaumegenthial.github.io/serving.html
# 最新作者的代码已经使用 tf.estimator与tf.data: https://github.com/guillaumegenthial/tf_ner

import numpy as np
import tensorflow as tf


from .data_utils import minibatches, pad_sequences, get_chunks
from .general_utils import Progbar, print_dict
from .base_model import BaseModel


class NERModel(BaseModel):
    """Specialized class of Model for NER"""

    def __init__(self, config):
        super(NERModel, self).__init__(config)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

        self.params = {}

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""
        # shape = (batch size, max length of sentence in batch)
        self.word_ids = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_ids")

        # shape = (batch size), batch中每个句子的长度(即每个句子有多少个词)
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                        name="sequence_lengths")

        # shape = (batch size, max length of sentence in batch, max length of word in batch)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, None, None],
                        name="char_ids")

        # shape = (batch_size, max_length of sentence in batch), batch中每个句子中每个词的长度
        self.word_lengths = tf.placeholder(tf.int32, shape=[None, None],
                        name="word_lengths")

        # shape = (batch size, max length of sentence in batch)
        self.labels = tf.placeholder(tf.int32, shape=[None, None],
                        name="labels")

        # hyper parameters
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                        name="dropout") # 此处dropout为keep_prob
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                        name="lr")


    def get_feed_dict(self, words, labels=None, lr=None, dropout=None):
        """Given some data, pad it and build a feed dictionary

        Args:
            words: list of sentences. A sentence is a list of ids of a list of
                words. A word is a list of ids
            labels: list of ids, 二维列表list, [batch, sentence_length]
            lr: (float) learning rate
            dropout: (float) keep prob

        Returns:
            dict {placeholder: value}

        """
        # perform padding of the given data
        if self.config.use_chars:
            # words=x_batch: [x=[char_ids=([6,9,1],[2,3,4],...,[5]), word_id=(18,20,15,...)],x=[...]]
            # labels=y_batch:[tag_id=[6,1,7,7,4],tag_id=[6,3,4,9],...]
            #
            # word: [char_ids=([6, 24, 9, 1], [2, 18, 24, 0, 0, 24], [23, 18, 3, 24, 4], [18, 1], [7, 24, 8], [15, 25, 0, 26], [5]),
            #         word_id=(18, 20, 15, 11, 12, 0, 2)]
            # label: tag_id=[6,1,7,7,4,3,7]
            # char_ids: (([6,24,9,1], [2,18,24,0,0,24]), ...)
            # word_ids: 二维列表[[6,5,...],[7,8,...]],此时传入的元素都是ids
            char_ids, word_ids = zip(*words)
            # word, paddding到此batch中最长句子的长度
            # sequence_lengths: 一维list, 此batch中每个句子padding前的长度
            # 每个句子padding到此batch中最长句子的长度
            word_ids, sequence_lengths = pad_sequences(word_ids, 0) # padding是在使用时才做,而并不存储
            # word_lengths: 二维list,此batch中每个句子每个单词的长度
            # 每个单词padding到此batch中最长单词的长度
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        else:
            word_ids, sequence_lengths = pad_sequences(words, 0)

        # build feed dictionary
        # 注意:key为placeholder
        feed = {
            self.word_ids: word_ids,
            self.sequence_lengths: sequence_lengths
        }

        if self.config.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = pad_sequences(labels, 0)
            feed[self.labels] = labels

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed, sequence_lengths


    def add_word_embeddings_op(self):
        """Defines self.word_embeddings

        If self.config.embeddings is not None and is a np array initialized
        with pre-trained word vectors, the word embeddings is just a look-up
        and we don't train the vectors. Otherwise, a random matrix with
        the correct shape is initialized.
        """
        with tf.variable_scope("words"):
            if self.config.embeddings is None:
                self.logger.info("WARNING: randomly initializing word vectors")
                # shape: vocab_word * word_dim
                _word_embeddings = tf.get_variable(
                        name="_word_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nwords, self.config.dim_word])
            else:
                # You should use tf.Variable with argument trainable=False instead of tf.constant,
                # otherwise you risk memory issues!
                _word_embeddings = tf.Variable(
                        self.config.embeddings, # glove vector:
                        name="_word_embeddings",
                        dtype=tf.float32,
                        trainable=self.config.train_embeddings)
            # word_ids: [batch size, max_max_sentence_len ]
            # word_embeddings: [ batch_size=8, max_sentence_len = 9, word_dim = 300]
            word_embeddings = tf.nn.embedding_lookup(_word_embeddings,
                    self.word_ids, name="word_embeddings")

        with tf.variable_scope("chars"):
            if self.config.use_chars:
                # get char embeddings matrix
                # shape : vocab_char * char_dim
                _char_embeddings = tf.get_variable(
                        name="_char_embeddings",
                        dtype=tf.float32,
                        shape=[self.config.nchars, self.config.dim_char])
                # char_ids: [ batch size, max length of sentence, max length of word ]
                # char_embeddings: [ batch_size=8, max_sentence_len = 9, max_word_len =9, char_dim= 100]
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                        self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                char_embeddings_shape = tf.shape(char_embeddings)

                self.params["char_embeddings"] = char_embeddings
                # shape:[ batch_size, max_sentence_len, max_word_len, char_dim]
                # new shape:[ batch_size*max_sentence_len, max_word_len, char_dim]
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[char_embeddings_shape[0]*char_embeddings_shape[1], char_embeddings_shape[-2], self.config.dim_char])
                # word_lengths: [batch_size, max_sentence_len]
                # new word_lengths: [batch_size*max_sentence_len]
                word_lengths = tf.reshape(self.word_lengths, shape=[char_embeddings_shape[0]*char_embeddings_shape[1]])

                # bi lstm on chars
                # state_is_tuple = True: accepted and returned states are 2-tuples of the c_state and m_state.
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_char,
                        state_is_tuple=True)
                # _output: ((output_fw, output_bw), (output_state_fw, output_state_bw))
                # output_fw: [batch_size(=8) * max_sentence_len(=9)=72, max_word_len(=9), hidden_size_char =100]
                # output_hidden_state_fw: [batch*max_sentence_len, hidden_size_char]
                #
                # ((output_fw, output_bw), (output_state_fw, output_state_bw)) = tf.nn.bidirectional_dynamic_rnn(...)
                # 其中output_state_fw是dynamic_rnn 的last_state输出,是(cell_state, hidden_state)的tuple
                # 即 output_state_fw = (cell_state, hidden_state) = (_, output_fw)
                # 通过调试运行如下：
                # _output:tuple(tuple( array(72,9,100),  # 正向output,所谓的output就是序列中所有的hidden_state输出值
                #                      array(72,9,100)), # 反向output
                #               tuple(lstmStateTuple(c=array(72,100), h= array(72,100)) , # 正向,state永远保持最近的一个,而不会记录所有历史
                #                     lstmStateTuple(c=array(72,100), h= array(72,100)) # 反向
                #               )
                # )
                # char_embeddings:[ batch_size*max_sentence_len, max_word_len, char_dim]
                # word_lengths: [batch_size*max_sentence_len],每个元素代表每个句子里有多少单词
                _output = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, char_embeddings,
                        sequence_length=word_lengths, # shape:[batch*max_sentence_len]
                        dtype=tf.float32,
                        time_major=False)
                self.params["bi_dynamic_rnn"] = _output
                # read and concat output

                # output_hidden_state_fw: [batch*max_sentence_len, hidden_size_char]
                # output_hidden_state_bw: [batch*max_sentence_len, hidden_size_char]
                (output_fw, output_bw), ((output_cell_state_fw, output_cell_state_bw), (output_hidden_state_fw, output_hidden_state_bw)) = _output
                # output: [batch_size*max_sentence_len, hidden_size_char*2]
                output = tf.concat([output_hidden_state_fw, output_hidden_state_bw], axis=-1)

                # new output: [batch size, max_sentence_len, hidden_size_char*2]
                output = tf.reshape(output,
                                    shape=[char_embeddings_shape[0], char_embeddings_shape[1], 2 * self.config.hidden_size_char])
                # 将word embedding与char embedding连接起来
                # word_embeddings: [ batch_size, max_sentence_len, word_dim]
                # char_embeddings: [ batch_size, max_sentence_len, hidden_size_char*2]
                # => [ batch_size, max_length_sentence, hidden_size_char*2 + word_dim],即 [8, 8, 100*2+300 = 500]
                word_embeddings = tf.concat([word_embeddings, output], axis=-1)
        # dropout
        # [ batch_size, max_sentence_len, hidden_size_char*2 + word_dim]
        self.word_embeddings =  tf.nn.dropout(word_embeddings, self.dropout)


    def add_logits_op(self):
        """Defines self.logits

        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            # word_embeddings: [ batch_size, max_sentence_len, hidden_size_char*2 + word_dim ]
            # sequence_lengths: [batch_size = 8], 如:[7,9,7,7,9,7,7,9]
            # output_fw: [ batch_size, max_sentence_len, hidden_size_lstm]
            # output_bw: [ batch_size, max_sentence_len, hidden_size_lstm]
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.word_embeddings,
                    sequence_length=self.sequence_lengths, dtype=tf.float32)
            # output: [ batch_size, max_sentence_len, hidden_size_lstm*2]
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout) # dropout:一般为keep prob,只在训练时存在

        with tf.variable_scope("proj"):
            # 将 embedding映射成tags
            # W: [self.config.hidden_size_lstm*2, ntags=9]
            W = tf.get_variable("W", dtype=tf.float32,
                    shape=[2*self.config.hidden_size_lstm, self.config.ntags])

            # b: [ntags]
            b = tf.get_variable("b", shape=[self.config.ntags],
                    dtype=tf.float32, initializer=tf.zeros_initializer())
            # nsteps: max_sentence_len
            nsteps = tf.shape(output)[1]
            # new output: [batch_size*max_sentence_len, hidden_size_lstm*2]
            output = tf.reshape(output, [-1, 2*self.config.hidden_size_lstm])
            # pred: [ batch_size*max_length_sentence, ntags]
            pred = tf.matmul(output, W) + b
            # logits: [batch_size, max_sentence_len, ntags]
            self.logits = tf.reshape(pred, [-1, nsteps, self.config.ntags])


    def add_pred_op(self):
        """Defines self.labels_pred

        This op is defined only in the case where we don't use a CRF since in
        that case we can make the prediction "in the graph" (thanks to tf
        functions in other words). With theCRF, as the inference is coded
        in python and not in pure tensroflow, we have to make the prediciton
        outside the graph.
        """
        if not self.config.use_crf:
            # labels_pred: [batch_size, max_sentence_len, ntags]
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                    tf.int32)


    def add_loss_op(self):
        """Defines the loss"""
        if self.config.use_crf:
            # logits: [batch_size, max_sentence_len, ntags], 注意logits值为各个tag的概率,
            # 此处用到了各个time_step的输出向量,最后在一起算总loss
            # labels: [batch size, max_sentence_len], labels为目标tag的index
            # log_likelihood: [batch_size=8],即输出为每个句子预测出该句的所有ner标签的log(Prob)之和
            # transition_params: [num_tags=9, num_tags=9]
            # sequence_lengths: [batch_size = 8], 如:[7,9,7,7,9,7,7,9]
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits, self.labels, self.sequence_lengths)
            self.trans_params = trans_params # need to evaluate it for decoding
            self.params["log_likelihood"] = log_likelihood
            self.params["trans_params"] = trans_params
            # loss: scalar, 我们希望log(Prob)最大,也即-log(Prob)最小
            self.loss = tf.reduce_mean(-log_likelihood) # 此处是对batch求平均
        else:
            # logits: [batch_size, max_sentence_len, ntags]
            # labels: [batch size, max_sentence_len]
            # losses: [batch_size, max_sentence_len]
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
            # sequence_lengths: [batch_size = 8], 如:[7,9,7,7,9,7,7,9]
            # mask: [batch_size=8, max_sentence_len=9], 值为bool的矩阵
            mask = tf.sequence_mask(self.sequence_lengths)
            self.params["mask"] = mask
            # new losses: [ num_of_word_count_in_batch ],
            # 它的shape为一个batch中所有句子的长度之和,每个元素为该句子的ner序列 log(Prob)
            losses_masked = tf.boolean_mask(losses, mask)
            self.params["losses_masked"] = losses_masked
            self.loss = tf.reduce_mean(losses_masked)

        # for tensorboard
        tf.summary.scalar("loss", self.loss)


    def build(self):
        # NER specific functions
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()

        # Generic functions that add training op and initialize session
        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                self.config.clip)
        self.initialize_session() # now self.sess is defined and vars are init


    def predict_batch(self, words):
        """
        Args:
            words: list of sentences

        Returns:
            labels_pred: list of labels for each sentence
            sequence_length

        """
        # 注意:测试阶段的dropout= 1.0
        fd, sequence_lengths = self.get_feed_dict(words, dropout=1.0)

        if self.config.use_crf:
            # get tag scores and transition params of CRF
            viterbi_sequences = []
            # logits: [batch_size, max_sentence_len, ntags], 注意logits值为各个tag的概率
            # transition_params: [num_tags=9, num_tags=9]
            logits, trans_params = self.sess.run(
                    [self.logits, self.trans_params], feed_dict=fd)

            # sequence_lengths: [batch_size = 8]
            # logits: [batch_size, max_sentence_len, ntags], 注意logits值为各个tag的概率
            # iterate over the sentences because no batching in vitervi_decode
            # 由于对batch进行遍历,所以batch = 1
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length] # keep only the valid steps
                # tf中的维特比解码无法使用batch模式,只能逐条解码
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        logit, trans_params)
                viterbi_sequences += [viterbi_seq]

            return viterbi_sequences, sequence_lengths

        else:
            labels_pred = self.sess.run(self.labels_pred, feed_dict=fd)

            return labels_pred, sequence_lengths


    def run_epoch(self, train, dev, epoch):
        """Performs one complete pass over the train set and evaluate on dev

        Args:
            train: dataset that yields tuple of sentences, tags
            dev: dataset
            epoch: (int) index of the current epoch

        Returns:
            f1: (python float), score to select model on, higher is better

        """
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = (len(train) + batch_size - 1) // batch_size
        prog = Progbar(target=nbatches)

        # iterate over dataset
        # words: list of zip obj, labels: 2维list
        for i, (words, labels) in enumerate(minibatches(train, batch_size)):
            fd, _ = self.get_feed_dict(words, labels, self.config.lr,
                    self.config.dropout)
        # max_sentence_len: 代表此batch中的最大句子长度,max_word_len:此batch中最长单词的长度
        # labels: [batch_size=8, max_sentence_len=9],2维list,注意：每个batch里的最长句子长度可能不一样, batch:8, max_sentence_len:每次不一样
        # word_ids: [batch_size = 8, max_sentence_len = 9] , 2维list, [8, 9]
        # sequence_lengths: [7, 9, 7, 7, 9, 7, 7, 9] , 表示每个句子的长度, 1维list
        # char_ids: [batch=8, max_sentence_len=9, max_word_len=9], 3维list,max_word_len为这个batch里最长的单词长度
        # word_lengths: [batch=8, max_sentence_len = 9]
            if self.config.debug_mode:
                _, train_loss, summary, params = self.sess.run(
                    [self.train_op, self.loss, self.merged, self.params], feed_dict=fd)
                if i == 0:
                    print_dict(params)
            else:
                _, train_loss, summary = self.sess.run(
                    [self.train_op, self.loss, self.merged], feed_dict=fd)
            # 更新进度条
            prog.update(i + 1, [("train loss", train_loss)])

            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch*nbatches + i)

        # 在开发集上进行验证
        metrics = self.run_evaluate(dev)
        msg = "dev set:"+" - ".join(["{} {:04.2f}".format(k, v)
                for k, v in metrics.items()])
        self.logger.info(msg)

        return metrics["f1"]


    def run_evaluate(self, test):
        """Evaluates performance on test set

        Args:
            test: dataset that yields tuple of (sentences, tags)

        Returns:
            metrics: (dict) metrics["acc"] = 98.4, ...

        """
        self.logger.info("Testing model over test set(ner model)")
        accs = []
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for words, labels in minibatches(test, self.config.batch_size):
            labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels, labels_pred,
                                             sequence_lengths):
                lab      = lab[:length]
                lab_pred = lab_pred[:length]
                # 这个准确率计算的是一个句子中所有标签的准确率
                accs    += [a==b for (a, b) in zip(lab, lab_pred)]

                # result = [("PER", 0, 2), ("LOC", 3, 4)]， 即(chunk_type, chunk_start, chunk_end)
                lab_chunks      = set(get_chunks(lab, self.config.vocab_tags))
                lab_pred_chunks = set(get_chunks(lab_pred, self.config.vocab_tags))

                correct_preds += len(lab_chunks & lab_pred_chunks)
                total_preds   += len(lab_pred_chunks)
                total_correct += len(lab_chunks)

        p   = correct_preds / total_preds if correct_preds > 0 else 0
        r   = correct_preds / total_correct if correct_preds > 0 else 0
        f1  = 2 * p * r / (p + r) if correct_preds > 0 else 0
        acc = np.mean(accs)

        return {"acc": 100*acc, "f1": 100*f1}


    def predict(self, words_raw):
        """Returns list of tags

        Args:
            words_raw: list of words (string), just one sentence (no batch)

        Returns:
            preds: list of tags (string), one for each word in the sentence

        """
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        pred_ids, _ = self.predict_batch([words])
        pred_tags = [self.idx_to_tag[idx] for idx in list(pred_ids[0])]

        return pred_tags
