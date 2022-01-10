# Built-in libraries
import time

# Third-party libraries
import mediapipe as mp
import cv2


#TODO: Fix bug with class initializing.
class HandTrackingDetector():
    
    def __init__(self, static_image_mode=False, model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_image_mode = static_image_mode,
        self.max_num_hands = max_num_hands,
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence,
        self.min_tracking_confidence = min_tracking_confidence
        
        self.media_pipe_hands = mp.solutions.hands
        self.media_pipe_draw = mp.solutions.drawing_utils  
        self.hands = self.media_pipe_hands.Hands(self.static_image_mode, self.max_num_hands, 
             self.model_complexity, self.min_detection_confidence, self.min_tracking_confidence)
    
    def hand_track(self, image, draw=True, highlight_base=False):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        # Verifies the presence of hands in the video to process them.
        if results.multi_hand_landmarks:
            for hands_lms in results.multi_hand_landmarks:
                # Draws the connections between the hand's nodes.
                if draw:
                    self.media_pipe_draw.draw_landmarks(image, hands_lms, self.media_pipe_hands.HAND_CONNECTIONS)
                
                # Highlights the base of the hands.
                if highlight_base:
                    for hand_id, lm in enumerate(hands_lms.landmark):
                        height, width, channel = image.shape
                        center_x, center_y = int(lm.x * width), int(lm.y * height)
                        
                        # Applies different mark for the base of the hand. 
                        if hand_id == 0:
                            cv2.circle(image, (center_x,center_y), 20, (255, 0, 0), cv2.FILLED)
            return image
                    


def main():
    # Initialize camera connection.
    capture = cv2.VideoCapture(0)

    # Time related variables
    previous_time = 0
    current_time = 0

    hand_detector = HandTrackingDetector()

    # Starts video capturing process.
    while True:
        success, img = capture.read()
        img = hand_detector.hand_track(image=img, draw=False, highlight_base=False)

        # FPS calculations and display.
        current_time = time.time()
        fps = 1/(current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 51, 51), 3)


        cv2.imshow('Image', img)
        cv2.waitKey(1)



if __name__ == '__main__':
    main()

