import os

from cnn_model import create_cnn_model, train_cnn_model, save_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm

def train_model(augmented_train_path, augmented_val_path, batch_size=32,
                epochs=50, learning_rate=0.001):

    # Set up data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        augmented_train_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        augmented_val_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Create and compile the model with GPU support
    input_shape = (224, 224, 3)
    num_classes = len(train_generator.class_indices)

    model = create_cnn_model(input_shape, num_classes, learning_rate=0.001)

    # Set up callbacks for training
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=False,
                                       monitor='val_loss', mode='min', verbose=1)
    tensorboard = TensorBoard(log_dir=os.path.join('logs', 'fit'), histogram_freq=1)

    # Train the model with GPU support and callbacks
    with tf.device('/GPU:0'):
        history = train_cnn_model(model, train_generator, validation_generator, epochs=epochs, batch_size=batch_size,
                                  callbacks=[model_checkpoint, tensorboard])

    # Save the trained model
    model_path = 'final_model.h5'
    save_model(model, model_path)

    return model_path


