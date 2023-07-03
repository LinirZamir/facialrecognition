import cv2
import dlib
from deepface import DeepFace

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

# Initialize the facial landmarks predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Predict facial landmarks for the detected face
        landmarks = predictor(gray, face)

        # Draw circles around the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

        # Extract the face ROI
        face_roi = frame[max(0, y1):min(y2, frame.shape[0]), max(0, x1):min(x2, frame.shape[1])]
        # Only proceed if the ROI is not empty
        if face_roi.size != 0:
            # Convert the face ROI to RGB (DeepFace expects RGB images)
            face_roi_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # Analyze the face ROI for emotion
            result = DeepFace.analyze(face_roi_rgb, actions=['emotion'], enforce_detection=False)

            if result:
                dominant_emotion = result[0]['dominant_emotion']
            else:
                dominant_emotion = "No emotion detected"

            # Display the dominant emotion
            cv2.putText(frame, dominant_emotion, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow("Frame", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
