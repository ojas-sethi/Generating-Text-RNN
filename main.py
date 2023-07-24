import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

text = text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQUENCE_LENGTH = 40
STEP = 3

# MODEL TRAINING STARTS
sentences = []
next_chars = []


for i in range(0, len(text) - SEQUENCE_LENGTH, STEP):
    sentences.append(text[i: i+SEQUENCE_LENGTH])
    next_chars.append(text[i+SEQUENCE_LENGTH])

x = np.zeros((len(sentences), SEQUENCE_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_chars[i]]] = 1



model = Sequential()
model.add(LSTM(128, input_shape=(SEQUENCE_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax')) 

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.01))
model.fit(x, y, batch_size=256, epochs=4)
model.save('textgenerator.model')
# MODEL TRAINING ENDS


model = tf.keras.models.load_model('textgenerator.model')

# This function is taken directly from the official Keras tutorial
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_ind = random.randint(0, len(text) - SEQUENCE_LENGTH - 1)
    generated = ''
    sentence = text[start_ind: start_ind + SEQUENCE_LENGTH]
    generated += sentence
    for i in range(length):
        x  = np.zeros((1, SEQUENCE_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

print('---------------Temperature: 0.2---------------')
print(generate_text(412, 0.2))

print('---------------Temperature: 0.4---------------')
print(generate_text(412, 0.4))

print('---------------Temperature: 0.6---------------')
print(generate_text(412, 0.6))

print('---------------Temperature: 0.7---------------')
print(generate_text(412, 0.7))

print('---------------Temperature: 0.8---------------')
print(generate_text(412, 0.8))

print('---------------Temperature: 0.9---------------')
print(generate_text(412, 0.9))

print('---------------Temperature: 1.0---------------')
print(generate_text(412, 1.0))
