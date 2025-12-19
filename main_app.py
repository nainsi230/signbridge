import os
import warnings
import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import tensorflow as tf
import google.generativeai as genai

# 1. SILENCE WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# 2. SETUP GOOGLE GEMINI
API_KEY = "AIzaSyAqw-pK9LjDvDKv5d8xP8Q1Asfdt69XXCs" # Your Key
try:
    genai.configure(api_key=API_KEY)
    # FIXED: Changed '2.5' to '1.5'
    ai_model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction="You are a Sign Language interpreter. Convert the user's broken English gloss into a natural, spoken sentence. Do not add intro text."
    )
    print("SUCCESS: Google Gemini connected!")
except Exception as e:
    print(f"WARNING: Google AI not connected. {e}")

# 3. SETUP TENSORFLOW MODEL
# We ONLY load the .keras model. We deleted the old pickle code.
try:
    model = tf.keras.models.load_model("asl_model_tf.keras")
    with open("label_encoder.p", "rb") as f:
        label_encoder = pickle.load(f)
    print("SUCCESS: TensorFlow Model Loaded!")
except Exception as e:
    print("CRITICAL ERROR: Could not load 'asl_model_tf.keras' or 'label_encoder.p'")
    print(f"Details: {e}")
    print("Did you run 'train_model_tf.py' yet?")
    exit()

# 4. MEDIA PIPE SETUP
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# UI Colors
COLOR_BG = (30, 30, 30)
COLOR_ACCENT = (0, 255, 217)
COLOR_TEXT = (255, 255, 255)
COLOR_BOX = (50, 50, 50)
COLOR_AI = (255, 100, 100)

# Variables
current_text = ""
ai_fixed_text = ""
last_prediction = ""
prediction_start_time = 0
CONFIDENCE_THRESHOLD = 0.85 # Increased slightly for stability
HOLD_TIME = 1.0
PAUSE_TIME = 2.0
last_hand_time = time.time()
space_added = False

cap = cv2.VideoCapture(0)

print("Starting Smart ASL Dashboard...")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    ui = np.zeros((720, 1280, 3), np.uint8)
    ui[:] = COLOR_BG 

    results = hands.process(rgb_frame)
    prediction = ""
    confidence = 0.0

    if results.multi_hand_landmarks:
        last_hand_time = time.time()
        space_added = False
        
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- EXTRACT FEATURES (3D) ---
            base_x = hand_landmarks.landmark[0].x
            base_y = hand_landmarks.landmark[0].y
            base_z = hand_landmarks.landmark[0].z

            xs, ys, zs = [], [], []
            for lm in hand_landmarks.landmark:
                xs.append(lm.x - base_x)
                ys.append(lm.y - base_y)
                zs.append(lm.z - base_z)
            
            # Normalize
            max_value = max(list(map(abs, xs + ys + zs)))
            if max_value != 0:
                xs = [x / max_value for x in xs]
                ys = [y / max_value for y in ys]
                zs = [z / max_value for z in zs]
            
            features = np.array([xs + ys + zs])
            
            try:
                # TensorFlow Prediction Logic
                predictions = model.predict(features, verbose=0) 
                prediction_idx = np.argmax(predictions)
                confidence = predictions[0][prediction_idx]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    prediction = label_encoder.inverse_transform([prediction_idx])[0]
                    
                    if prediction == last_prediction:
                        if (time.time() - prediction_start_time) > HOLD_TIME:
                            if prediction == "SPACE":
                                current_text += " "
                            elif prediction == "DEL":
                                current_text = current_text[:-1]
                            else:
                                current_text += prediction
                            
                            prediction_start_time = time.time()
                            last_prediction = ""
                            ai_fixed_text = "" 
                    else:
                        last_prediction = prediction
                        prediction_start_time = time.time()
            except Exception as e:
                # This prints the error if the model fails, instead of hiding it
                print(f"Prediction Error: {e}")

    else:
        if not space_added and (time.time() - last_hand_time) > PAUSE_TIME and current_text != "":
            current_text += " "
            space_added = True
            last_prediction = ""

    # --- UI DRAWING ---
    x_offset = 320
    y_offset = 100
    # Center the camera
    resized_frame = cv2.resize(frame, (640, 480))
    ui[y_offset:y_offset+480, x_offset:x_offset+640] = resized_frame
    cv2.rectangle(ui, (x_offset-5, y_offset-5), (x_offset+640+5, y_offset+480+5), COLOR_ACCENT, 4)

    # Info Panel
    cv2.putText(ui, "DETECTED:", (1000, 200), cv2.FONT_HERSHEY_PLAIN, 2, COLOR_TEXT, 2)
    if prediction != "":
        cv2.putText(ui, prediction, (1020, 350), cv2.FONT_HERSHEY_SIMPLEX, 5, COLOR_ACCENT, 10)
        cv2.putText(ui, f"{int(confidence*100)}%", (1000, 450), cv2.FONT_HERSHEY_PLAIN, 1.5, COLOR_TEXT, 1)

    # Text Boxes
    cv2.putText(ui, "RAW GESTURES:", (200, 590), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_ACCENT, 1)
    cv2.rectangle(ui, (200, 600), (680, 680), COLOR_BOX, cv2.FILLED)
    cv2.putText(ui, current_text[-20:], (210, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_TEXT, 2)

    cv2.putText(ui, "GOOGLE GEMINI FIX (Press Enter):", (700, 590), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_AI, 1)
    cv2.rectangle(ui, (700, 600), (1180, 680), (60, 40, 40), cv2.FILLED) 
    if ai_fixed_text:
         cv2.putText(ui, ai_fixed_text, (710, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_AI, 2)
    else:
         cv2.putText(ui, "Waiting...", (710, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100,100,100), 1)

    # Keyboard
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == 13: # Enter
        if current_text.strip() != "":
            print("Sending to Gemini...")
            try:
                prompt = f"Convert this sign language gloss to a normal, fluent English sentence: '{current_text}'"
                response = ai_model.generate_content(prompt)
                ai_fixed_text = response.text.strip()
                print("Gemini says:", ai_fixed_text)
            except Exception as e:
                print("Gemini Error:", e)
                ai_fixed_text = "Connection Error"
    elif key == 8: # Backspace
        current_text = current_text[:-1]

    cv2.imshow("Smart ASL Dashboard", ui)

cap.release()
cv2.destroyAllWindows()