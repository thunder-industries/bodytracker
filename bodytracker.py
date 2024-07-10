import cv2
import mediapipe as mp

def main():
    cap = cv2.VideoCapture(0)  # Open the default camera
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces_results = face_detection.process(frame_rgb)

            # Draw face detection results
            if faces_results.detections:
                for detection in faces_results.detections:
                    mp_drawing.draw_detection(frame, detection)

            # Detect hands
            frame_rgb = cv2.flip(frame_rgb, 1)  # Flip horizontally for selfie-view display
            hands_results = hands.process(frame_rgb)

            # Draw hand detection results
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Body Tracker', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()