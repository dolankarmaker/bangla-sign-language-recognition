import tensorflow as tf
from train import train_model
from inference import inference_on_video, inference_on_live_stream
from text_to_speech import translate_to_bengali, text_to_speech, play_sound


def set_gpu_device(device='/GPU:0'):
    # Set the GPU device for TensorFlow operations
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            tf.config.experimental.set_visible_devices(physical_devices[0], device)
            print(f"Using GPU: {physical_devices[0]}")
        except RuntimeError as e:
            print(e)


if __name__ == "__main__":
    # Set the GPU device
    set_gpu_device('/GPU:0')  # Specify the GPU index if you have multiple GPUs

    # Define paths
    original_dataset_path = 'D:\\Bangla Sign Lanugage Recognition\\Dataset_Sign_Language'
    augmented_dataset_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language'
    augmented_train_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language\\train'
    augmented_val_path = 'D:\\Bangla Sign Lanugage Recognition\\Augmented_Dataset_Sign_Language\\validation'
    test_video_path = 'test\\video.mp4'

    # Training
    trained_model_path = train_model(original_dataset_path, augmented_dataset_path, augmented_train_path,
                                     augmented_val_path, learning_rate=0.001)

    # Inference
    recognized_signs = inference_on_video(trained_model_path, test_video_path)

    # Translate each recognized sign to Bengali
    bengali_signs = [translate_to_bengali(sign) for sign in recognized_signs]

    # Convert each translated sign to speech and play it
    for sign in bengali_signs:
        tts = text_to_speech(sign, lang='bn')
        play_sound(tts)
