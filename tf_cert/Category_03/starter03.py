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
# Computer Vision with CNNs
#
# Build a classifier for Rock-Paper-Scissors based on the rock_paper_scissors
# TensorFlow dataset.
#
# IMPORTANT: Your final layer should be as shown. Do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail.
#
# NOTE THAT THIS IS UNLABELLED DATA.
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()


    TRAINING_DIR = "tmp/rps/"
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=5,
        zoom_range=0.1,
        shear_range=0.7,
        fill_mode='nearest',
        validation_split=0.25)

    train_generator = training_datagen.flow_from_directory(
        "tmp/rps/",
        target_size=(150, 150),
        batch_size=2000,
        class_mode='categorical',
        shuffle=True,
        subset='training')

    test_generator = training_datagen.flow_from_directory(
        "tmp/rps/",
        target_size=(150, 150),
        batch_size=2000,
        class_mode='categorical',
        shuffle=True,
        subset='validation')

    x_train = train_generator[0][0]
    y_train = train_generator[0][1]
    x_test = test_generator[0][0]
    y_test = test_generator[0][1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu'),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(2, 2), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor='val_acc', patience=10, mode='auto', verbose=1)

    hist = model.fit(x_train, y_train, epochs=500,
                     callbacks=[es],
                     validation_split=0.05,
                     steps_per_epoch=32,
                     validation_steps=1)

    acc = hist.history['acc']
    val_acc = hist.history['val_acc']

    loss = model.evaluate(x_test, y_test)
    print('acc : ', acc[-10])
    print('val_acc : ', val_acc[-10])
    print('loss : ', loss[0])
    return model

    '''
    acc :  1.0
    val_acc :  0.8947368264198303
    loss :  1.0329103469848633
    '''

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
