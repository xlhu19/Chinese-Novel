#!/usr/bin/python3
#coding:utf-8

import os
import pickle
import re

def load_text(path):
    input_file = os.path.join(path)
    with open(input_file, 'r') as f:
        text_data = f.read()
    return text_data

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    print("===== vocab")
    print(vocab)
    
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

def token_lookup():
    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]
    return dict(zip(symbols, tokens))

def load_data():
    return pickle.load(open('preprocess.p', mode='rb'))

if __name__ == '__main__':
    num_words_for_training = 100000
    
    dir = './data/寒门首辅.txt'
    text = load_text(dir)
    text = text[:num_words_for_training]
    lines_of_text = text.split('\n')
    
    lines_of_text = lines_of_text[14:]
    lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]
    
    lines_of_text = [lines.strip() for lines in lines_of_text]
    
    pattern = re.compile(r'\[.*\]')
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    
    pattern = re.compile(r'<.*>')
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    
    pattern = re.compile(r'\.+')
    lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]
    
    pattern = re.compile(r' +')
    lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]
    
    pattern = re.compile(r'\\r')
    lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]
    
    print(lines_of_text[0])
    print("===== lines %d" % (len(lines_of_text)))

    token_dict = token_lookup()

    for key, token in token_dict.items():
        text = text.replace(key, '{}'.format(token))
    print(text)
    text = list(text)

    vocab_to_int, int_to_vocab = create_lookup_tables(text)
    int_text = [vocab_to_int[word] for word in text]

    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

