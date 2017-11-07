#!/usr/bin/python3
#coding:utf-8

import problem_unittests as tests
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import numpy as np
import preprocess
from tensorflow.contrib import seq2seq

def check():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.2'), 'Please use TensorFlow version 1.2 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        print('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def get_init_cell(batch_size, rnn_size):
    num_layers = 2
        
    keep_prob = 0.8
    
    cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    
    drop = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
    cell = tf.contrib.rnn.MultiRNNCell([drop for _ in range(num_layers)])
    
    init_state = cell.zero_state(batch_size, tf.float32)
    
    init_state = tf.identity(init_state, name='init_state')

    return cell, init_state

def get_embed(input_data, vocab_size, embed_dim):
    
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim)), dtype=tf.float32)
    
    return tf.nn.embedding_lookup(embedding, input_data)

def build_rnn(cell, inputs):
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name="final_state")
    return outputs, final_state

def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    embed = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embed)
    
    # remember to initialize weights and biases, or the loss will stuck at a very high point
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None,
                                               weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
                                               biases_initializer=tf.zeros_initializer())
    
    return logits, final_state

def get_batches(int_text, batch_size, seq_length):
    
    n_batches = (len(int_text) // (batch_size * seq_length))

    batch_origin = np.array(int_text[: n_batches * batch_size * seq_length])
    batch_shifted = np.array(int_text[1: n_batches * batch_size * seq_length + 1])
    
    batch_shifted[-1] = batch_origin[0]
    
    batch_origin_reshape = np.split(batch_origin.reshape(batch_size, -1), n_batches, 1)
    batch_shifted_reshape = np.split(batch_shifted.reshape(batch_size, -1), n_batches, 1)

    batches = np.array(list(zip(batch_origin_reshape, batch_shifted_reshape)))
    
    return batches


if __name__ == '__main__':
    check()
    int_text, vocab_to_int, int_to_vocab, token_dict = preprocess.load_data()
    
    num_epochs = 300
    batch_size = 256
    rnn_size = 512
    embed_dim = 512
    seq_length = 30
    learning_rate = 0.003
    show_every_n_batches = 30
    
    save_dir = './save'
    
    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
    
        input_data_shape = tf.shape(input_text)
        print(input_data_shape)
    
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
    
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
    
        probs = tf.nn.softmax(logits, name='probs')
    
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))
    
        optimizer = tf.train.AdamOptimizer(lr)
    
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
    
    batches = get_batches(int_text, batch_size, seq_length)
    
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})
    
            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)
    
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))
    
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')
    
    pickle.dump((seq_length, save_dir), open('params.p', 'wb'))

