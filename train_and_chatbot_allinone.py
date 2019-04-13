import pickle
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout
from keras.layers import add, dot, concatenate
from keras.layers import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

with open("train_qa.txt", "rb") as fp:   # Unpickling
    train_data =  pickle.load(fp)

with open("test_qa.txt", "rb") as fp:   # Unpickling
    test_data =  pickle.load(fp)

##### make vocab, add yes and no

vocab = set()

all_data = test_data + train_data

for story, question , answer in all_data:
    vocab = vocab.union(set(story))
    vocab = vocab.union(set(question))
vocab.add('no')
vocab.add('yes')

#extra space for keras pad seq
vocab_len = len(vocab) + 1

max_story_len = max([len(data[0]) for data in all_data])
max_question_len = max([len(data[1]) for data in all_data])

########### vectorizing the Data

# Reserve 0 for pad_sequences
vocab_size = len(vocab) + 1

# integer encode sequences of words
tokenizer = Tokenizer(filters=[])
tokenizer.fit_on_texts(vocab)

train_story_text = []
train_question_text = []
train_answers = []

for story,qs,answer in train_data:
    train_story_text.append(story)
    train_question_text.append(question)

train_story_seq = tokenizer.texts_to_sequences(train_story_text)

len(train_story_text)

len(train_story_seq)

### function for vectorisation 
## x = words, xq = qs, y = ans. returns tuple for unpacking
def vectorize_stories(data, word_index=tokenizer.word_index, max_story_len=max_story_len,max_question_len=max_question_len):

    X = []
    Xq = []
    Y = []


    for story, query, answer in data:

        x = [word_index[word.lower()] for word in story]
        xq = [word_index[word.lower()] for word in query]
        y = np.zeros(len(word_index) + 1)

        y[word_index[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)


    return (pad_sequences(X, maxlen=max_story_len),pad_sequences(Xq, maxlen=max_question_len), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_data)

inputs_test, queries_test, answers_test = vectorize_stories(test_data)

################## The Model

input_sequence = Input((max_story_len,))
question = Input((max_question_len,))


## encoders for building the networks

### input encoder m

#embedding
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size,output_dim=64))
input_encoder_m.add(Dropout(0.3))

# op: samples, story_maxlen, embedding_dim

### input encoder c

# embed the input into a sequence of vectors of size query_maxlen
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size,output_dim=max_question_len))
input_encoder_c.add(Dropout(0.3))
# op: samples,story_maxlen, query_maxlen

### qs encoder

# embed the question into a sequence of vectors
question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size,
                               output_dim=64,
                               input_length=max_question_len))
question_encoder.add(Dropout(0.3))
# output: (samples, query_maxlen, embedding_dim)

### encode the sequences

# encode input sequence and questions (which are indices)
# to sequences of dense vectors
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

#####  dot product to compute the match between first input vector seq and the query

# shape: `(samples, story_maxlen, query_maxlen)`
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation('softmax')(match)

####  this match matrix with the second input vector sequence

# add the match matrix with the second input vector sequence
response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)

#concat
answer = concatenate([response, question_encoded])

answer = LSTM(32)(answer)  # (samples, 32)

answer = Dropout(0.5)(answer)
answer = Dense(vocab_size)(answer)  # (samples, vocab_size)

#softmax to get the probabiolity weightings
answer = Activation('softmax')(answer)

#finalise
model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
epoch_no = 150
batch_size = 32
history = model.fit([inputs_train, queries_train], answers_train,batch_size=batch_size,epochs=epoch_no,validation_data=([inputs_test, queries_test], answers_test))

filename =f'{epoch_no)_epoch_{batch_size}.h5' 
model.save(filename)

model.load_weights(filename)
pred_results = model.predict(([inputs_test, queries_test]))

print(test_data[0][0])

story =' '.join(word for word in test_data[0][0])
print(story)

query = ' '.join(word for word in test_data[0][1])
print(query)

print("true ans",test_data[0][2])

#Generate prediction from model
val_max = np.argmax(pred_results[0])

for key, val in tokenizer.word_index.items():
    if val == val_max:
        k = key

print("pred ans: ", k)
print("prob certainty: ", pred_results[0][val_max])


######Begin 'chat bot interface'. 

######NB: due to current (documented) errors with model.load() in new sessions, i am keeping the chat bot
######functionality here until i can find a workaround to have it in its own file (the original plan is to have 
######all the code training the model in one file, save the model to json and the weights to .h5, then reload
#####for QNA chat bot application 

def check_valid(input):
    for word in input.split():
        if not (word in vocab):
            print (f'"{word}" is not in the vocab list."\n')
            return -1


def ask_story():
    print('input story: ')
    x = input()
    if (x == 1):
        print(vocab)
        return ask_story()
    elif check_valid(x) == -1:
        print('not a valid story')
        return ask_story()
    else:
        return x

def ask_qs():
    print('input question: ')
    x = input()
    if (x == 1):
        print(vocab)
        return ask_qs()
    elif check_valid(x) == -1:
        return ask_qs()
    else:
        return x



print('Model training complete\n')


def menu():
    print('\nPlease input your own story using only words from the vocab list. Please leave a space either side of full',
          'stops and question marks.\n',
          'Type "1" for the list of available words. Type 0 to exit.\n')

    global my_story
    my_story = ask_story()
    print('\n')
    global my_question
    print('\n')
    my_question = ask_qs()

    data = [(my_story.split(), my_question.split(), 'yes')]

    # print(f"This is the data: {data}")

    story, qs, answer = vectorize_stories(data)
    pred_results = model.predict(([story, qs]))

    val_max = np.argmax(pred_results[0])

    for key, val in tokenizer.word_index.items():
       if val == val_max:
           k = key

    print("Predicted answer is: ", k)
    print("Probability of certainty was: ", pred_results[0][val_max])
    return menu()



menu()









