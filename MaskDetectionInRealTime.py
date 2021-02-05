import cv2
import tensorflow
from tensorflow.keras.models import load_model

detector = load_model(r'C:/Users/ashis/Downloads/dummy.model')

import tensorflow as tf
import cv2
import numpy

# starting the video stream
cap = cv2.VideoCapture(0)

# using the XML file for haarcascade classifier
classifier = cv2.CascadeClassifier(r"C:/Users/ashis/Downloads/haarcascade_frontalface_default.xml")

# using the loop to watch the stream in real time.
while True:
    (success, frame) = cap.read()  # reading the frame from the stream
    new_image = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0] // 1))  # resizing the frame to speed up the process of detection
    face = classifier.detectMultiScale(new_image)  # detecting faces from the frame(ROI)
    for x, y, w, h in face:
        try:
            face_img = new_image[y:x + h, x:x + w]  # getting the coordinates for the face detected
            resized = cv2.resize(face_img,(224, 224))  # resizing the  face detected to fit into the model in the shape(224,224)
            image_array = tf.keras.preprocessing.image.img_to_array(resized)  # converting the detected image into an array
            image_array = tf.expand_dims(image_array, 0)  # expanding the dimensions to fit in the model
            predictions = detector.predict(image_array)  # making predictions on the ROI
            score = tf.nn.softmax(predictions[0])  # getting the results
            label = numpy.argmax(score)
        except Exception as e:
            print('bad frame')

        if label == 0:
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(new_image, "mask", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif label == 1:
            cv2.rectangle(new_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(new_image, 'no_mask', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            None
    # displaying the window after predicting the outcome
    cv2.imshow('face_window', new_image)
    print(numpy.argmax(score), 100 * numpy.max(score))
    # waitkey to terminate the loop
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
# release the stream
cap.release()
cv2.destroyAllWindows()
