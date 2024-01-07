from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import tensorflow as tf
import tqdm

def create_cnn_model(input_shape, num_classes, learning_rate=0.001):
    # This function creates a CNN model based on the provided input shape and number of classes
    model = models.Sequential()

    # Add convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected layers with Dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # Adjust the optimizer and learning rate

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def plot_training_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def train_cnn_model(model, train_generator, validation_generator, epochs=20, batch_size=32, callbacks=None):
    with tf.device('/GPU:0'):  # Specify the GPU index if you have multiple GPUs
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            callbacks=callbacks,  # Pass the callbacks argument
            verbose=1,  # Set verbose to 1 to enable the default progress bar
        )

    return history

def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")


def load_model(model_path):
    loaded_model = models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return loaded_model
