#-*- coding: utf-8 -*-

#RNN for text classification
#@author: Littlely
#@time: 2018/2/13

import tensorflow as tf
import sklearn
import numpy as np
import pandas as pd
import math
import jieba
import pickle
import time
from collections import Counter

class RNNTextClassifier():
    def __init__(self,vocab_size, n_out, embedding_size=128, cell_size=128,
                 grad_clip=5.0,sess=tf.Session()):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.cell_size = cell_size
        self.grad_clip = grad_clip
        self.n_out = n_out
        self.sess = sess
        self._pointer = None
        self.buildgraph()

    def buildgraph(self):
        self.add_input_layer()
        self.add_wordembedding_layer()
        self.add_dynamic_rnn()
        self.add_output_layer()
        self.add_optimizer()

    def add_input_layer(self,):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int64, [None])
        self.X_seq_len = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        self.lr = tf.placeholder(tf.float32)
        self._pointer = self.X

    def add_wordembedding_layer(self):
        embedding = tf.get_variable("encoder",
                                    [self.vocab_size,self.embedding_size],
                                    dtype=tf.float32,
                                    initializer=tf.random_uniform_initializer(-1.0,1.0))
        embedded = tf.nn.embedding_lookup(embedding, self._pointer)
        # self._pointer = tf.nn.dropout(embedded, keep_prob=self.keep_prob)
        self._pointer = embedded

    def lstm_cell(self):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.cell_size,initializer=tf.orthogonal_initializer())
        return tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob= self.keep_prob)

    def add_dynamic_rnn(self):
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell(),
            inputs=self._pointer,
            sequence_length=self.X_seq_len,
            dtype=tf.float32
        )
    def add_output_layer(self):
        self.logits = tf.layers.dense(self.last_state.h, self.n_out)

    def add_optimizer(self):
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.Y
            )
        )
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, axis=1),self.Y),dtype=tf.float32))
        #gradient clipping
        params = tf.trainable_variables()
        gradients = tf.gradients(ys=self.loss, xs=params)
        clipped_gradients, _ = tf.clip_by_global_norm(t_list=gradients, clip_norm=self.grad_clip)
        self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(clipped_gradients, params))

    def fit(self, X, Y, val_data=None, n_epoch=10, batch_size=128, exp_decay=True,
            isshuffle=True, keep_prob=0.5):
        if val_data is None:
            print("Train %d samples" % len(X))
        else:
            print("Train %d samples | Test %d samples" % (len(X), len(val_data[0])))
        log = {'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
        global_step = 0
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for epoch in range(n_epoch):
            if isshuffle:
                X, Y = sklearn.utils.shuffle(X,Y)
            for local_step, ((X_batch, X_batch_lens), Y_batch) in enumerate(
                    zip(self.next_batch(X, batch_size), self.gen_batch(Y, batch_size))):
                lr = self.decrease_lr(exp_decay,global_step, n_epoch, len(X), batch_size)
                _, loss, acc = self.sess.run([self.train_op, self.loss, self.acc],
                                             feed_dict={self.X:X_batch,
                                                        self.Y:Y_batch,
                                                        self.X_seq_len:X_batch_lens,
                                                        self.lr:lr,
                                                        self.keep_prob:keep_prob})
                global_step += 1
                if local_step % 50 == 0:
                    print("Epoch %d | Step %d%d | Train loss: %.4f | Train acc: %.4f | lr: %.4f" % (
                        epoch+1, local_step, int(len(X)/batch_size), loss, acc, lr
                    ))
                log['loss'].append(loss)
                log['acc'].append(acc)

            if val_data is not None:
                val_loss_list, val_acc_list = [],[]
                for (X_test_batch,X_test_batch_lens), Y_test_batch in zip(self.next_batch(val_data[0], batch_size),
                                                                          self.gen_batch(val_data[1],batch_size)):
                    v_loss, v_acc = self.sess.run([self.loss, self.acc],feed_dict={
                        self.X: X_test_batch, self.Y: Y_test_batch,
                        self.X_seq_len:X_test_batch_lens, self.keep_prob:1.0
                    })
                    val_loss_list.append(v_loss)
                    val_acc_list.append(v_acc)
                val_loss, val_acc = self.list_avg(val_loss_list), self.list_avg(val_acc_list)
                log['val_loss'].append(val_loss)
                log['val_acc'].append(val_acc)
                print("val_data loss: %.4f | val_data acc: %.4f" % (val_loss, val_acc))
        saver.save(self.sess,"c:/users/ll/desktop/model/model.ckpt") #your save path
        return log

    def predict(self, X_test, batch_size=128):
        batch_pred_list = []
        for (X_test_batch, X_test_batch_lens) in self.next_batch(X_test, batch_size):
            batch_pred = self.sess.run(self.logits,feed_dict={
                self.X: X_test_batch,
                self.X_seq_len: X_test_batch_lens,
                self.keep_prob: 1.0
            })
            batch_pred_list.append(batch_pred)
        return np.argmax(np.vstack(batch_pred_list), 1)

    def pad_sentence_batch(self, sentence_batch, pad_int=0):
        max_lens = max([len(sentence) for sentence in sentence_batch])
        padded_seqs = []
        seq_lens = []
        for sentence in sentence_batch:
            padded_seqs.append(sentence + [pad_int] * (max_lens-len(sentence)))
            seq_lens.append(len(sentence))

        return padded_seqs, seq_lens

    def next_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            padded_seqs, seq_lens = self.pad_sentence_batch(arr[i:i+batch_size])
            yield padded_seqs, seq_lens

    def gen_batch(self, arr, batch_size):
        for i in range(0, len(arr), batch_size):
            yield arr[i: i+batch_size]

    def list_avg(self, l):
        return sum(l)/len(l)

    def decrease_lr(self, exp_decay, global_step, n_epoch, len_x, batch_size):
        if exp_decay:
            max_lr = 0.005
            min_lr = 0.001
            decay_rate = math.log(min_lr/max_lr) / (-n_epoch*len_x/batch_size)
            lr = max_lr*math.exp(-decay_rate*global_step)
        else:
            lr = 0.001
        return lr


def load_data(file_in_path, pickle_text=True, pickle_out_path=None):
    '''
    file_in_path=".../input/training.csv"
    pickle_out_path=".../output/texting.txt"
    '''
    train_data = pd.read_csv(file_in_path, header=None, names=['ind', 'text'])
    texts = train_data['text']
    ind = train_data['ind']
    ind = np.asarray(ind)
    text1 = []
    for text in texts:
        text1.append(" ".join(jieba.cut(text)))
    text1 = [s.split(" ") for s in text1]
    if pickle_text:
        if pickle_out_path is not None:
            dictionary = {'ind':ind, 'texts': text1}
            with open(pickle_out_path, "wb") as f:
                pickle.dump(dictionary, f)
        else:
            print("you should provide pickle_out_path")

    return ind, text1

def add_dict(texts,row_key_word=5, limits=3000):
    countlist = []
    dictionary = set()
    word2index = dict()
    for text in texts:
        countlist.append(Counter(text))
    for count in countlist:
        tfidf = dict()
        for word in count:
            tfidf[word] = _tf(word, count) * _idf(word, countlist)
        sorted_word = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:row_key_word]
        word = [w[0] for w in sorted_word]
        for w in word:
            dictionary.add(w)
        if len(dictionary) > limits+1:
            break
    for i, word in enumerate(dictionary):
        word2index[word] = i+1 #need add the unknown word, index 0
    word2index['UNK'] = 0
    return word2index

def convert_text(texts,row_key_word=5, limits=20000, ispickle=False, pickle_out_path=None):
    textlist = []
    word2index = add_dict(texts, row_key_word, limits)
    for text in texts:
        wordlist = []
        for word in text:
            if word in word2index:
                wordlist.append(word2index[word])
            else:
                wordlist.append(word2index["UNK"])
        textlist.append(wordlist)
    if ispickle is not None:
        with open(pickle_out_path, 'wb') as f:
            pickle.dump(textlist, f)
    return textlist

def _tf(word, count):
    return count[word] / sum(count.values())

def _containing(word, countlist):
    return sum(1 for count in countlist if word in count)

def _idf(word,countlist):
    return math.log(len(countlist)/(1+_containing(word, countlist)))


# ind, texts = load_data(".../input/training.csv", True, ".../output/text.txt")
# id, test_texts = load_data(".../input/testing.csv",True, ".../output/test_texts.txt")
with open(".../output/text.txt", 'rb') as f:
    train_data = pickle.load(f)
with open(".../output/test_texts.txt", 'rb') as f:
    test_data = pickle.load(f)

ind, train_texts = train_data['ind'], train_data['texts']
ind -= 1
_, test_texts = test_data['ind'], test_data['texts']

# textlist_train = convert_text(train_texts, row_key_word=7, limits=10000,
#                         ispickle=True, pickle_out_path=".../output/textlist_train.txt")
# textlist_test = convert_text(test_texts, row_key_word=7, limits=10000,
#                         ispickle=True, pickle_out_path=".../output/textlist_test.txt")
# print(textlist_train[:2])
with open(".../output/textlist_train.txt", 'rb') as f:
    textlist_train = pickle.load(f)
with open(".../output/textlist_test.txt", 'rb') as f:
    textlist_test = pickle.load(f)

rnn = RNNTextClassifier(10004, 11)
t1 = time.time()
log = rnn.fit(textlist_train,ind)
t2 = time.time() - t1
print(t2)
result = rnn.predict(textlist_test)
result = result + 1
t3 = time.time() - t2
print(t3)
print("training time: %f, testing time: %f" % (t2, t3))
print(result)
result = pd.DataFrame(list(result))
result.to_csv(".../output/result.csv", sep=",")
