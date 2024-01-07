import os
import cv2
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img


# def save_image(img, save_path):
#     cv2.imwrite(save_path, img)

def save_image(img_array, img_path):
    # Ensure values are in the valid range [0, 1] before saving
    img_array = np.clip(img_array, 0.0, 1.0)
    plt.imshow(img_array)
    plt.axis('off')
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0, format='jpg', dpi=300)
    plt.close()


def augment_and_save_images(original_dataset_path, augmented_dataset_path, augmentation_params, num_augmentations=5):
    print("At data augmentation")
    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path)

    class_directories = [d for d in os.listdir(original_dataset_path) if
                         os.path.isdir(os.path.join(original_dataset_path, d))]
    augmentation_datagen = ImageDataGenerator(**augmentation_params)

    # Loop through each class directory
    for class_dir in class_directories:
        current_class_augmented_path = os.path.join(augmented_dataset_path, class_dir)
        os.makedirs(current_class_augmented_path, exist_ok=True)

        image_files = [f for f in os.listdir(os.path.join(original_dataset_path, class_dir)) if f.endswith('.jpg')]

        for image_file in image_files:
            img_path = os.path.join(original_dataset_path, class_dir, image_file)
            img = load_and_preprocess_image(img_path)
            img = np.expand_dims(img, axis=0)
            augmented_images = augmentation_datagen.flow(img, batch_size=1)
            for i in range(num_augmentations):
                augmented_img = augmented_images.next()[0]
                # print(augmented_img)
                base_name = os.path.basename(img_path)
                file_name, file_extension = os.path.splitext(base_name)
                augmented_img_path = os.path.join(current_class_augmented_path, f'{file_name}_{i}{file_extension}')
                save_image(augmented_img, augmented_img_path)


def split_augmented_dataset(augmented_dataset_path, augmented_train_path, augmented_val_path, test_size=0.2,
                            random_state=42):
    class_directories = [d for d in os.listdir(augmented_dataset_path) if
                         os.path.isdir(os.path.join(augmented_dataset_path, d))]

    # Split the augmented dataset into training and validation sets
    for class_dir in class_directories:
        class_images = os.listdir(os.path.join(augmented_dataset_path, class_dir))
        train_images, val_images = train_test_split(class_images, test_size=test_size, random_state=random_state)

        # Create directories for each class in the training and validation sets
        os.makedirs(os.path.join(augmented_train_path, class_dir), exist_ok=True)
        os.chmod(os.path.join(augmented_train_path, class_dir), 0o777)
        os.makedirs(os.path.join(augmented_val_path, class_dir), exist_ok=True)
        os.chmod(os.path.join(augmented_train_path, class_dir), 0o777)

        # Copy images to the training set
        for img in train_images:
            src_path = os.path.join(augmented_dataset_path, class_dir, img)
            dest_path = os.path.join(augmented_train_path, class_dir, img)
            # Set read and write permissions for the user
            os.chmod(src_path, 0o600)  # Replace source_path with the actual path
            # Set read, write, and execute permissions for the user
            os.chmod(dest_path, 0o700)  # Replace destination_directory with the actual path
            shutil.copy(src_path, dest_path)

        # Copy images to the validation set
        for img in val_images:
            src_path = os.path.join(augmented_dataset_path, class_dir, img)
            dest_path = os.path.join(augmented_val_path, class_dir, img)
            # Set read and write permissions for the user
            os.chmod(src_path, 0o600)  # Replace source_path with the actual path
            # Set read, write, and execute permissions for the user
            os.chmod(dest_path, 0o700)  # Replace destination_directory with the actual path
            shutil.copy(src_path, dest_path)


original_dataset_path = 'D:\\Bangla Sign Lanugage Recognition\\Dataset_Sign_Language'
augmented_dataset_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language'
augmented_train_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language\\train'
augmented_val_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language\\validation'

augmentation_params = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

augment_and_save_images(original_dataset_path, augmented_dataset_path, augmentation_params, num_augmentations=5)
split_augmented_dataset(augmented_dataset_path, augmented_train_path, augmented_val_path, test_size=0.2,
                        random_state=42)
