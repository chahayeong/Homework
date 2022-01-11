# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json') as file:
        data = json.load(file)

    dataset = []
    for elem in data:
        sentences.append(elem['headline'])
        labels.append(elem['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    from tensorflow.keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    train_padded = pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
    validation_padded = pad_sequences(validation_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)



    model = tf.keras.Sequential([
    # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam'
                  , metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

    import time

    start_time = time.time()
    hist = model.fit(train_padded, train_labels, epochs=1000, batch_size=512, verbose=2,
              callbacks=[es], validation_data=(validation_padded, validation_labels))
    end_time = time.time() - start_time

    val_loss = hist.history['val_loss']
    acc = hist.history['acc']

    print("time : ", end_time)
    print('val_loss : ', val_loss[-20])
    print('acc : ', acc[-20])
    '''
    time :  13.322257995605469
    val_loss :  0.42744195461273193
    acc :  0.8754000067710876
    '''

    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
