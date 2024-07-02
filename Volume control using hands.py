import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Video capture using OpenCV
cap = cv2.VideoCapture(0)

# Variable to store previous distance between thumb and index finger
prev_distance = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for natural movement perception
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for the thumb and index finger
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]

            # Calculate distance between thumb and index finger
            distance = calculate_distance(thumb_x, thumb_y, index_x, index_y)

            # Set initial previous distance
            if prev_distance is None:
                prev_distance = distance

            # Volume control logic
            sensitivity = 0.5  # Adjust sensitivity as needed
            volume_change = (distance - prev_distance) * sensitivity

            # Change system volume based on hand movement
            current_volume = pyautogui.volume()
            new_volume = current_volume + volume_change

            # Ensure volume stays within valid range (0 to 100)
            new_volume = max(0, min(100, new_volume))
            pyautogui.moveTo(100, 100, new_volume/100)

            # Update previous distance
            prev_distance = distance

    # Display the frame with landmarks
    cv2.imshow('Hand Tracking', frame)

    # Exit the program by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
