import keras as k
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# Initializing the CNN (convolutional neural network)
classifier = Sequential()

# Adding convolutional layers and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Fully connected layers (Dense layers)
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', 
                   loss='binary_crossentropy', 
                   metrics=['accuracy'])

# Fitting the CNN to the images
train_data = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

training_set = train_data.flow_from_directory(r'C:\Users\Priyanshu Kumar\Desktop\ML AI\train',
                                              target_size=(64, 64),
                                              batch_size=32,
                                              class_mode='binary')

test_data = ImageDataGenerator(rescale=1./255)

testing_set = test_data.flow_from_directory(r'C:\Users\Priyanshu Kumar\Desktop\ML AI\test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Training the model using `fit()` instead of `fit_generator()`
classifier.fit(training_set,
               steps_per_epoch=4000,
               epochs=10,
               validation_data=testing_set,
               validation_steps=1000)

# Making a prediction on a new image
# test_image = image.load_img(r'C:\Users\Priyanshu Kumar\Desktop\ML AI\dogorcat', target_size=(64, 64))
test_image = image.load_img(r'C:\Users\Priyanshu Kumar\Desktop\ML AI\dogorcat2.jpg', target_size=(64, 64))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
