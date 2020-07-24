import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

train_dataset = pd.read_csv('train.csv')
test_dataset = pd.read_csv('test.csv')
train_dataset.head()

# Data Preprocessing
X = train_dataset.drop(columns='label')
y = train_dataset['label']

X = X.values.reshape(-1, int(np.sqrt(784)), int(np.sqrt(784)), 1)
test_dataset = test_dataset.values.reshape(-1, int(np.sqrt(784)), int(np.sqrt(784)), 1)

# Showing the images
rows = 5
cols = 6
counter = 0
fig = plt.figure(figsize=(15,7))
for i in range(1, rows*cols+1):
    fig.add_subplot(rows, cols, i)
    plt.imshow(np.squeeze(X[counter + i-1]), cmap='gray')
    plt.title(y[counter + i-1], fontsize=16)
    plt.axis(False)
    fig.add_subplot
counter += rows*cols

y = to_categorical(y, num_classes=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1455)

# Model Architecture
model = Sequential()
input_shape = (28, 28, 1)

model.add(Conv2D(32, (5, 5), input_shape = input_shape, activation='relu', padding='same'))
model.add(Conv2D(32, (5, 5), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

adam  = Adam(learning_rate = 0.00001)
model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=23,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
datagen.fit(X_train)

train_history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = 512),
                                    validation_data = (X_test, y_test), verbose = 2,
                                    epochs=30, steps_per_epoch=X_train.shape[0]//512)

y_pred = model.predict(test_dataset)
y_pred = np.argmax(y_pred, axis=1)

output = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})
output.to_csv('MNIST_prediction.csv', index=False)

model.save('MNIST_model.h5')



