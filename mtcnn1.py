import cv2
import numpy as np
from keras.models import model_from_json
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the MTCNN face detector
detector = MTCNN()

# Load the emotion model
json_file = open('E:/Nandani_D22180/Emotion_detection_with_CNN-main/Emotion_detection_with_CNN-main/model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("E:/Nandani_D22180/Emotion_detection_with_CNN-main/Emotion_detection_with_CNN-main/model/emotion_model.h5")
print("Loaded model from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('E:/Nandani_D22180/DIP/DIP_FINAL_PROJECT_D22180_S22043/Results/EMOCNN_IMAGES_page-0001 (1).jpg')
#"C:\Users\NANDANI\Downloads\pexels-pavel-danilyuk-8626760-1920x1080-25fps.mp4"
#cap = cv2.VideoCapture('C:/Users/NANDANI/Downloads/pexels-pavel-danilyuk-8627834-1080x1920-25fps.mp4')
while True:
    # Read the video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_frame)

    # Process each detected face for emotion detection
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Preprocess the face for emotion detection
        roi_gray_frame = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion labels
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Exit on ESC key press
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close any open windows
cap.release()
cv2.destroyAllWindows()
