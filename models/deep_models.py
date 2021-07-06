"""
TensorFlow implementation of DeepModels(FM, DNN, DEEPFM, etc)
by PennLee
"""
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tensorflow.layers import batch_normalization

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self, feature_size, field_size,
                 use_fm=True, use_deep=True,
                 embedding_size=7,
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5], deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=False, batch_norm_decay=0.995,
                 random_seed=2021,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.2):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.epoch = epoch
        self.batch_size = batch_size

        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.train_result, self.valid_result = [], []

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(self.random_seed)
            # ---------- 数据和参数定义 ----------
            # 喂入模型的数据定义
            self.X_index = tf.placeholder(tf.int32, shape=[None, self.field_size], name="X_index")  # None * F
            self.X_value = tf.placeholder(tf.float32, shape=[None, self.field_size], name="X_value")  # None * F
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name="y")  # None * 1
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            # 模型参数定义
            self.weights = self._initialize_model_weights()

            # 每个样本选择需要迭代的embedding参数
            self.embeddings_lookup = tf.nn.embedding_lookup(self.weights["embeddings"], self.X_index)  # None * F * K
            X_value = tf.reshape(self.X_value, shape=[-1, self.field_size, 1])
            self.embeddings = tf.multiply(self.embeddings_lookup, X_value)

            # ---------- FM部分 ----------
            # FM的一阶项和
            self.w_order1 = tf.nn.embedding_lookup(self.weights["W"], self.X_index)  # None * F * 1
            self.y_order1 = tf.reduce_sum(tf.add(tf.multiply(self.w_order1, X_value), self.weights['b']), 1)  # None * 1

            # 和方项sum(vx) * sum(vx)
            self.sum_vx_square = tf.square(tf.reduce_sum(self.embeddings, 1))  # None * K
            # 方和项sum(vx * vx)
            self.square_sum_vx = tf.reduce_sum(tf.square(self.embeddings), 1)  # None * K
            # FM的二阶项和
            self.y_order2 = 0.5 * tf.reduce_sum(tf.subtract(self.sum_vx_square, self.square_sum_vx), axis=1, keepdims=True)  # None * 1

            # ---------- Deep部分 ----------
            dropout_keep_deep = tf.cond(self.train_phase, lambda: self.dropout_deep, lambda: [1.0] * len(self.dropout_deep))
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights[f"W_layer_{i}"]), self.weights[f"b_layer_{i}"]) # None * layer[i] * 1
                if self.batch_norm:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn=f"bn_{i}") # None * layer[i] * 1
                self.y_deep = self.deep_layers_activation(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, dropout_keep_deep[i+1])  # dropout at each Deep layer
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["W_layer_out"]), self.weights["b_layer_out"])

            # ---------- DeepFM ----------
            self.y_hat = self.y_order1  # LR
            if self.use_fm:
                self.y_hat = self.y_order1 + self.y_order2  # FM
                if self.use_deep:
                    self.y_hat = self.y_order1 + self.y_order2 + self.y_deep  # DeepFM

            # ---------- 损失和正则 ----------
            # 损失
            if self.loss_type == "logloss":
                self.y_hat = tf.nn.sigmoid(self.y_hat)
                self.loss = tf.losses.log_loss(self.y, self.y_hat)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.y_hat))

            # l2正则
            if self.l2_reg >0:
                self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["W"])
                if self.use_fm:
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["embeddings"])
                    if self.use_deep:
                        for i in range(len(self.deep_layers)):
                            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights[f"W_layer_{i}"])
                        self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["W_layer_out"])

            # ---------- 优化方法 ----------
            # optimizer
            if self.optimizer_type == "adam":
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == "adagrad":
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == "gd":
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == "momentum":
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = self._init_session()
            self.sess.run(init)

    def _initialize_model_weights(self):
        weights = dict()
        # 一阶参数定义 —— LR
        weights['W'] = tf.Variable(tf.random_uniform([self.feature_size, 1], 0.0, 1), name='W')  # feature_size * 1
        weights['b'] = tf.Variable([0.0])
        # embedding参数定义
        weights['embeddings'] = tf.Variable(tf.random_normal([self.feature_size, self.embedding_size], 0.0, 0.01),
                                            name='embeddings')
        # DNN参数定义
        layer_cnt = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        for i in range(layer_cnt):
            if i == 0:
                he = np.sqrt(2.0 / input_size)
                w_size = (input_size, self.deep_layers[0])
            else:
                he = np.sqrt(2.0 / self.deep_layers[i-1])
                w_size = (self.deep_layers[i-1], self.deep_layers[i])
            weights[f"W_layer_{i}"] = tf.Variable(np.random.normal(loc=0, scale=he, size=w_size),
                                                  dtype=np.float32, name=f"W_layer_{i}")
            weights[f"b_layer_{i}"] = tf.Variable(tf.zeros((1, self.deep_layers[i])), dtype=np.float32, name=f"bias_layer_{i}")
        # DNN最后一层
        he = np.sqrt(2.0 / self.deep_layers[-1])
        weights["W_layer_out"] = tf.Variable(np.random.normal(loc=0, scale=he, size=(self.deep_layers[-1], 1)),
                                                   dtype=np.float32, name="W_layer_out")
        weights["b_layer_out"] = tf.Variable(tf.constant(0.0), dtype=np.float32, name=f"bias_layer_out")
        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_normalization(x, center=True, scale=True, training=True, name=scope_bn)
        bn_inference = batch_normalization(x, center=True, scale=True, training=False, reuse=True, name=scope_bn)
        bn = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return bn

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)

    def get_batch(self, Xi, Xv, y, batch_size, i):
        start, end = i * batch_size, (i + 1) * batch_size
        return Xi[start: end], Xv[start: end], [[l] for l in y[start: end]]

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, y_valid=None,
            early_stopping_rounds=None, greater_is_better=True, verbose=0):
        """
        Xi_train: 训练集特征索引, shape=[[I1], [I2]...]
        Xv_train: 训练集特征值, shape=[[V1], [V2]...]
        y_train: 训练集标签, shape=[y1, y2....]
        Xi_valid: 验证集特征索引
        Xv_valid: 验证集特征值
        y_valid: 验证集标签
        """
        try:
            sess.close()
        except:
            pass
        self._init_graph()
        had_valid = Xi_valid is not None
        for epoch in range(1, self.epoch + 1):
            for i in range(int(len(y_train) / self.batch_size) + 1):
                Xi_batch, Xv_batch, y_batch = self.get_batch(Xi_train, Xv_train, y_train, self.batch_size, i)
                feed_dict = {self.X_index: Xi_batch,
                             self.X_value: Xv_batch,
                             self.y: y_batch,
                             self.train_phase: True}
                _ = self.sess.run(self.optimizer, feed_dict=feed_dict)

            curr_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # 训练集评估
            ev_result_train = self.evaluate(y_train, self.predict(Xi_train, Xv_train))
            self.train_result.append({'epoch': epoch, 'curr_time': curr_time, 'train_result': ev_result_train})
            # 验证集评估
            if had_valid:
                ev_result_valid = self.evaluate(y_valid, self.predict(Xi_valid, Xv_valid))
                self.valid_result.append({'epoch': epoch, 'curr_time': curr_time,
                                          'train_result': ev_result_train, 'valid_result': ev_result_valid})
            # 打印评测指标
            if verbose and epoch % verbose == 0:
                if had_valid:
                    print(self.log_info(self.valid_result[-1]))
                else:
                    print(self.log_info(self.train_result[-1], has_valid=False))

            # early_stopping
            if verbose and had_valid and (early_stopping_rounds is not None) and \
                    self.early_stop(self.valid_result, early_stopping_rounds, greater_is_better):
                print('Stopping. Best iteration:')
                print(self.log_info(self.valid_result[-early_stopping_rounds]))
                break

    def predict(self, Xi, Xv):
        feed_dict = {self.X_index: Xi,
                     self.X_value: Xv,
                     self.train_phase: False}
        pred = self.sess.run(self.y_hat, feed_dict=feed_dict)
        return pred.reshape((len(pred), ))

    def evaluate(self, label, pred):
        return self.eval_metric(label, pred)

    def early_stop(self, valid_result, early_stopping_rounds=50, greater_is_better=True):
        # TODO: 动态存储当前最后结果的模型
        if len(valid_result) > early_stopping_rounds:
            last_rounds_result = [result['valid_result'] for result in valid_result[-early_stopping_rounds:]]
            if not greater_is_better:
                return all([result > last_rounds_result[0] for result in last_rounds_result[1:]])
            else:
                return all([result < last_rounds_result[0] for result in last_rounds_result[1:]])
        return False

    def log_info(self, ev_result, has_valid=True):
        if has_valid:
            return "[%d] [%s] train-result=%.4f, valid-result=%.4f " % (
                ev_result['epoch'], ev_result['curr_time'], ev_result['train_result'], ev_result['valid_result'])
        return "[%d] [%s] train-result=%.4f" % (ev_result['epoch'], ev_result['curr_time'], ev_result['train_result'])
