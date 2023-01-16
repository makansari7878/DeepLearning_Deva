import json

import keras.layers
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras import Sequential
import numpy as np

training_size = 20000
vocalbulary_size = 10000
data = [json.loads(line) for line in open('Sarcasm_Headlines_Dataset.json', 'r', encoding='utf-8')]
#print(data)

sentences = []
lables = []

for item in data:
    sentences.append(item['headline'])
    lables.append(item['is_sarcastic'])

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = lables[0:training_size]
testing_labels = lables[training_size:]

tonkenizer = Tokenizer(num_words=vocalbulary_size, oov_token='<OOV>')
tonkenizer.fit_on_texts(training_sentences)

word_index = tonkenizer.word_index
#print(word_index)

training_sequences = tonkenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=100,padding='post', truncating='post')

testing_sentences_sequences = tonkenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences_sequences, maxlen=100,padding='post', truncating='post')


training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

#---------------------------------------------------------------------------

model = Sequential([
    keras.layers.Embedding(vocalbulary_size, 16, input_length=100 ),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

result = model.fit(training_padded, training_labels, epochs=20,
                  validation_data=(testing_padded, testing_labels), verbose=2 )


testing_input = ['boehner just wants wife to listen, not come up with alternative debt-reduction ideas',
                 "the 'roseanne' revival catches up to our thorny political mood, for better and worse"]
test_seq = tonkenizer.texts_to_sequences(testing_input)
test_padded = pad_sequences(test_seq, maxlen=100, padding='post', truncating='post')
print(model.predict(test_padded))