import cv2
import numpy as np
import mediapipe as mp_google
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe_utils as mp

class HandTrackerWebcam:
    def __init__(self, solo=True, pd_score_thresh=0.5, lm_score_thresh=0.5, use_gesture=True, **kwargs):
        self.solo = solo
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        self.use_gesture = use_gesture
        
        # Enforce 1152 x 648 frame size as expected by mouse_controller.py
        self.img_w = 1152
        self.img_h = 648
        self.use_lm = True
        self.video_fps = 30
        
        # Open the built-in webcam (index 0)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open built-in webcam. Please ensure a webcam is connected and accessible.")
            
        # Initialize Google MediaPipe Tasks HandLandmarker
        base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1 if solo else 2,
            min_hand_detection_confidence=pd_score_thresh,
            min_hand_presence_confidence=lm_score_thresh,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, [], None
            
        # Resize to 1152 x 648
        frame = cv2.resize(frame, (self.img_w, self.img_h))
        # Flip the frame horizontally for a natural mirror-like webcam view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp_google.Image(image_format=mp_google.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        hands = []
        if results.hand_landmarks and results.handedness:
            for idx, hand_landmarks in enumerate(results.hand_landmarks):
                # Access handedness
                category = results.handedness[idx][0]
                label = category.category_name.lower() # "left" or "right"
                score = category.score
                
                # Instantiate a HandRegion object using the project's utility class
                hand = mp.HandRegion()
                hand.lm_score = score
                hand.label = label
                
                # Extract landmarks
                norm_lms = []
                pixel_lms = []
                for lm in hand_landmarks:
                    norm_lms.append([lm.x, lm.y, lm.z])
                    pixel_lms.append([int(lm.x * self.img_w), int(lm.y * self.img_h)])
                
                hand.norm_landmarks = np.array(norm_lms)
                hand.landmarks = np.array(pixel_lms)
                
                # Compute bounding box size for dynamic sizing in renderer
                xs = hand.landmarks[:, 0]
                ys = hand.landmarks[:, 1]
                rect_w = max(xs) - min(xs)
                rect_h = max(ys) - min(ys)
                hand.rect_w_a = max(rect_w, rect_h)
                
                # Run gesture recognizer using project's utilities
                if self.use_gesture:
                    mp.recognize_gesture(hand)
                    
                hands.append(hand)
                
        return frame, hands, None

    def exit(self):
        self.cap.release()
        self.detector.close()
