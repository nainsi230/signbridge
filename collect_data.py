import cv2
import mediapipe as mp
import numpy as np
import csv
import os

# Define the letters you want to train
# Added 'SPACE' and 'DEL' for hackathon demo utility
CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "SPACE", "DEL"]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

DATA_FILE = "hand_data_3d.csv" # Changed name to avoid mixing with old 2D data

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header: label, then x0..x20, y0..y20, z0..z20
        header = ["label"] + \
                 [f"x{i}" for i in range(21)] + \
                 [f"y{i}" for i in range(21)] + \
                 [f"z{i}" for i in range(21)]
        writer.writerow(header)

cap = cv2.VideoCapture(0)

print("INSTRUCTIONS:")
print("1. Press 'space' key for SPACE gesture (maybe an open palm?)")
print("2. Press 'backspace' key for DELETE gesture (maybe a closed fist moving left?)")
print("3. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- NEW 3D LOGIC ---
            # Wrist is the anchor point (index 0)
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z # Depth anchor
            
            xs, ys, zs = [], [], []
            
            for lm in hand_landmarks.landmark:
                xs.append(lm.x - base_x)
                ys.append(lm.y - base_y)
                zs.append(lm.z - base_z) # Relative depth
            
            # Normalize Scale (Optional but recommended for Hackathons)
            # This helps if you move your hand closer/further from camera
            # We calculate the max distance from wrist and divide all points by it
            max_value = max(list(map(abs, xs + ys + zs)))
            if max_value != 0:
                xs = [x / max_value for x in xs]
                ys = [y / max_value for y in ys]
                zs = [z / max_value for z in zs]

            # Combine x, y, z
            data_row = xs + ys + zs

            # Check for key presses
            key = cv2.waitKey(1)
            if key != -1:
                char = ""
                if key == 32: # Space bar
                    char = "SPACE"
                elif key == 8: # Backspace
                    char = "DEL"
                else:
                    try:
                        char = chr(key).upper()
                    except:
                        pass
                
                if char in CLASSES:
                    with open(DATA_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([char] + data_row)
                    print(f"Saved 3D sample for: {char}")

    cv2.imshow("3D Data Collector", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()