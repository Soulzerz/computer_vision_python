# Built-in libraries
import time

# Third-party libraries
import mediapipe as mp
import cv2


# Initialize camera connection.
capture = cv2.VideoCapture(0)

media_pipe_hands = mp.solutions.hands
media_pipe_draw = mp.solutions.drawing_utils

# Time related variables
previous_time = 0
current_time = 0

# Parameters -> static_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
hands = media_pipe_hands.Hands()


# Starts video capturing process.
while True:
    success, img = capture.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    # Verifies the presence of hands in the video to process them.
    if results.multi_hand_landmarks:
        for hands_lms in results.multi_hand_landmarks:
            for hand_id, lm in enumerate(hands_lms.landmark):
                height, width, channel = img.shape
                center_x, center_y = int(lm.x * width), int(lm.y * height)
                
                # Applies different mark for the base of the hand. 
                if hand_id == 0:
                    cv2.circle(img, (center_x,center_y), 20, (255, 0, 0), cv2.FILLED)
                
            
            media_pipe_draw.draw_landmarks(img, hands_lms, media_pipe_hands.HAND_CONNECTIONS)

    # FPS calculations and display.
    current_time = time.time()
    fps = 1/(current_time-previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 51, 51), 3)


    cv2.imshow('Image', img)
    cv2.waitKey(1)
