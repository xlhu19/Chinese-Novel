#coding:utf-8

import helper
import re

dir = './data/寒门首辅.txt'
text = helper.load_text(dir)

num_words_for_training = 100000

text = text[:num_words_for_training]

lines_of_text = text.split('\n')

print(len(lines_of_text))

print(lines_of_text[:15])

lines_of_text = lines_of_text[14:]

print(lines_of_text[:5])

lines_of_text = [lines for lines in lines_of_text if len(lines) > 0]

print(len(lines_of_text))

print(lines_of_text[:20])

lines_of_text = [lines.strip() for lines in lines_of_text]

print(lines_of_text[:20])

# 生成一个正则，负责找『[]』包含的内容
pattern = re.compile(r'\[.*\]')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

print(lines_of_text[:20])

# 将上面的正则换成负责找『<>』包含的内容
pattern = re.compile(r'<.*>')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

# 将上面的正则换成负责找『......』包含的内容
pattern = re.compile(r'\.+')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("。", lines) for lines in lines_of_text]

print(lines_of_text[:20])

# 将上面的正则换成负责找行中的空格
pattern = re.compile(r' +')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("，", lines) for lines in lines_of_text]

print(lines_of_text[:20])

print(lines_of_text[-20:])

# 将上面的正则换成负责找句尾『\\r』的内容
pattern = re.compile(r'\\r')

# 将所有指定内容替换成空
lines_of_text = [pattern.sub("", lines) for lines in lines_of_text]

print(lines_of_text[-20:])

print(len(lines_of_text))

def create_lookup_tables(input_data):
    
    vocab = set(input_data)
    
    # 文字到数字的映射
    vocab_to_int = {word: idx for idx, word in enumerate(vocab)}
    
    # 数字到文字的映射
    int_to_vocab = dict(enumerate(vocab))
    
    return vocab_to_int, int_to_vocab

def token_lookup():

    symbols = set(['。', '，', '“', "”", '；', '！', '？', '（', '）', '——', '\n'])
    
    tokens = ["P", "C", "Q", "T", "S", "E", "M", "I", "O", "D", "R"]

    return dict(zip(symbols, tokens))

helper.preprocess_and_save_data(''.join(lines_of_text), token_lookup, create_lookup_tables)

