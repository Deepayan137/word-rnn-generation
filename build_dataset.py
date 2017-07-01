import json
import argparse
import os
import re


def tokenize(iota):
    punctuations = ' ,."\'/%()\n;:!-?'
    tokens = re.split('[%s]'%(punctuations), iota)
    return tokens

text = open('alice_wonderland.txt', 'r').read()
all_words = tokenize(text)
all_words = (' '.join(all_words)).split()
unique_words = list(set(all_words))
vocab_size = len(unique_words)
seq_length = 30
if os.path.exists('vocabulary.json') == False:
	print("Building vocabulary ......")
	word_to_int = dict((w, i) for i, w in enumerate(unique_words))
	with open('vocabulary.json', 'w') as json_file:
		json.dump(word_to_int, json_file, indent=4)




def build_dataset():
	X = []
	y=[]
	with open('vocabulary.json') as json_data:
		word_to_int = json.load(json_data)
		
	for i in range(0, len(all_words) - seq_length):
		seq_in = all_words[i: i + seq_length]
		seq_out = all_words[i + seq_length]
		X.append(list(map(lambda x: word_to_int[x], seq_in)))
		y.append(word_to_int[seq_out])		


	return X, y, vocab_size

