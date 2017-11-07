#!/usr/bin/python3
#coding:utf-8

import tensorflow as tf
import numpy as np
import problem_unittests as tests
import pickle

# np.set_printoptions(threshold='nan')

_, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
seq_length, load_dir = pickle.load(open('params.p', mode='rb'))

def get_tensors(loaded_graph):
   
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    
    initial_state = loaded_graph.get_tensor_by_name("init_state:0")
    
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    
    probs = loaded_graph.get_tensor_by_name("probs:0")
    
    return inputs, initial_state, final_state, probs

def pick_word(probabilities, int_to_vocab):
   
    ret = []
   
    base = 0.05
    print("len(probabilities): " + str(len(probabilities)))
    for idx, prob in enumerate(probabilities):
        if prob >= base:
            print(prob)
            base = prob
            ret.append(int_to_vocab[idx])
    
    return str(ret[-1])

gen_length = 150

prime_word = '章'
print(prime_word)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

    gen_sentences = [prime_word]
    prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

    for n in range(gen_length):
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state})
        
        pred_word = pick_word(probabilities[0][dyn_seq_length - 1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    novel = ''.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '（', '“'] else ''
        novel = novel.replace(token.lower(), key)
    novel = novel.replace('\n ', '\n')
    novel = novel.replace('（ ', '（')
        
    print(novel)
