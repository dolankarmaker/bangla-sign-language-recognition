import cv2
import numpy as np
import tensorflow as tf
from cnn_model import load_model


def preprocess_frame(frame, target_size=(224, 224)):
    resized_frame = cv2.resize(frame, target_size)
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)
    return preprocessed_frame


def map_class_to_sign(class_index):
    # Replace the following mapping with your specific sign classes
    sign_mapping = {
        0: 'Bad',
        1: 'Beautiful',
        2: 'Friend',
        3: 'Good',
        4: 'House',
        5: 'Me',
        6: 'My',
        7: 'Request',
        8: 'Skin',
        9: 'Urine',
        10: 'You'
    }

    return sign_mapping.get(class_index, 'Unknown Sign')


def inference_on_video(saved_model_path, video_path):
    model = load_model(saved_model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Define a list to store recognized signs
    recognized_signs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)  # You need to implement this function

        # Convert the preprocessed frame to the input format expected by the model
        input_data = np.expand_dims(preprocessed_frame, axis=0)

        # Make predictions
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions)

        # Map the predicted class to the corresponding sign (you need to implement this mapping)
        recognized_sign = map_class_to_sign(predicted_class)

        # Display the recognized sign on the frame
        cv2.putText(frame, recognized_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Add the recognized sign to the list
        recognized_signs.append(recognized_sign)

        # Display the frame
        cv2.imshow('Sign Language Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

    # Return the recognized signs
    return recognized_signs


def inference_on_live_stream(model):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = preprocess_frame(frame)
        processed_frame = np.expand_dims(processed_frame, axis=0)

        # Make predictions
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions[0])

        # Display the result on the frame
        cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Inference on Live Stream', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage for inference
    model_path = 'D:\\Bangla Sign Lanugage Recognition\\best_model.h5'  # Replace with the path to your trained model
    loaded_model = load_model(model_path)

    # Example inference on video
    video_path_to_test = 'test//video.mp4'
    inference_on_video(loaded_model, video_path_to_test)

    # Example inference on live stream
    inference_on_live_stream(loaded_model)
