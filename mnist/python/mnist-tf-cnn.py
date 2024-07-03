import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_cnn():
    model = Sequential(name="mnist-bpga-cnn")
    model.add(InputLayer(input_shape=(28, 28, 1)))
    model.add(Conv2D(6, (5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu', padding='valid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(120, (4, 4), activation='relu', padding='valid'))
    model.add(Flatten())
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255

# split the training set into training and validation sets
validation_images = train_images[-10000:]
validation_labels = train_labels[-10000:]
train_images = train_images[:-10000]
train_labels = train_labels[:-10000]

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(train_images, train_labels, batch_size=30)
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=30)

# train
cnn_model = create_cnn()
cnn_model.compile(optimizer=SGD(lr=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_model.fit(train_generator, epochs=80, validation_data=validation_generator)

# evaluate
test_loss, test_accuracy = cnn_model.evaluate(test_images, test_labels, batch_size=30)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')
