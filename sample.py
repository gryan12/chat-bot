import pickle
import numpy as np


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
