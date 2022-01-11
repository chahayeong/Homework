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
# Basic Datasets Question
#
# Create a classifier for the Fashion MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the Fashion MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#
import tensorflow as tf


def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # YOUR CODE HERE
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    print(x_train.shape, y_train.shape)

    from sklearn.preprocessing import OneHotEncoder
    one = OneHotEncoder()
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    one.fit(y_train)
    y_train = one.transform(y_train).toarray()  # (60000, 10)
    y_test = one.transform(y_test).toarray()  # (10000, 10)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=2, padding='same',
                     activation='relu', input_shape=(28, 28)))
    model.add(tf.keras.layers.Conv1D(32, 2, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool1D())
    model.add(tf.keras.layers.Conv1D(64, 2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Conv1D(64, 2, padding='same', activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(124, activation='relu'))
    model.add(tf.keras.layers.Dense(84, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='acc')

    hist = model.fit(x_train, y_train, epochs=10000, batch_size=576, verbose=2,
                     validation_split=0.005, callbacks=[es])
    acc = hist.history['acc']
    loss = model.evaluate(x_test, y_test)
    print('acc =', acc[-10])
    print('loss =', loss)

    '''
    acc = 0.9520267844200134
    loss = [0.576040506362915, 0.8716999888420105]
    '''
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
