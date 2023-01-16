from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.utils import pad_sequences

data = "My name is Ansari and  my city name is bangalore"
# result = text_to_word_sequence(data)
# print(result)
#
# result = set(text_to_word_sequence(data))
# print(result)
# vocbulary = len(result)
# print(vocbulary)

data1 = ["My name is Ansari and ", " my city name is bangalore", "Allah is mercyful"]
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(data1)

# tokenizer.fit_on_texts(data1)
# index = tokenizer.word_index
# print(index)
#
# print(tokenizer.word_counts)
# print(tokenizer.document_count)
# print(tokenizer.word_docs)

seq = tokenizer.texts_to_sequences(data1)
print(seq)

# test_data = ['Allah is greatest', "Allah is the most Gracious"]
# test_seq = tokenizer.texts_to_sequences(test_data)
# print(test_seq)

#padding
test_data = ['Allah is greatest', "Allah is the most Gracious","Allah"]
test_seq = tokenizer.texts_to_sequences(test_data)
padding_res = pad_sequences(test_seq, padding='post')
print(padding_res)


