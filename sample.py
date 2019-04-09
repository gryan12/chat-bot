import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

#data

with open('train_qa.txt', 'rb') as f: 
     train_data = pickle.load(f)

with open('test_qa.txt','rb') as f: 
    test_data = pickle.load(f)

#print(for_training)
print(type(test_data))
print(type(train_data))
print(len(test_data))
print(len(train_data))
print(train_data[0])
print(' '.join(train_data[0][0]))
print(' '.join(test_data[1][1]))
vocab = set()
all_data = test_data + train_data
for story, question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))

vocab.add('no')
vocab.add('yes')
print(vocab)
vocab_len = len(vocab) + 1
max_story_len = max([len(data[0]) for data in all_data])

tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

tokenizer.word_index



