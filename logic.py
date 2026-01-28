import cv2
import mediapipe as mp
import math
import time
import pyautogui
from collections import deque

class GestureController:
    def __init__(self):
        # MediaPipe Setup
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_mesh
        self.mp_draw = mp.solutions.drawing_utils

        # Initialize Models
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # =====================================================
        # TIMING & COOLDOWN CONFIGURATION (The Fix)
        # =====================================================
        self.active = False
        
        # 1. Post-Action Cooldown: How long to wait AFTER an action (prevents spam)
        self.last_action_time = 0
        self.POST_ACTION_COOLDOWN = 2.0  # Increased to 2 seconds for safety

        # 2. Pre-Action Hold: How long to HOLD a gesture before it counts
        # This fixes the "transition" issue. You must hold the pose for 0.6s.
        self.HOLD_DURATION = 0.6 
        self.current_gesture = None
        self.gesture_start_time = 0

        # Blink Variables
        self.blink_timestamps = []
        self.EAR_THRESHOLD = 0.22
        self.BLINK_WINDOW = 3.0

        # Pinch Config
        self.last_pinch_x = None
        self.PINCH_THRESHOLD = 0.04
        self.PINCH_MOVE_THRESHOLD = 0.03

        # Indices
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def _euclid(self, p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_ear(self, landmarks, eye_indices):
        a = self._euclid(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        b = self._euclid(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        c = self._euclid(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        return (a + b) / (2.0 * c)

    def _count_fingers(self, landmarks, label):
        lm = landmarks.landmark
        count = 0
        if label == "Right":
            if lm[4].x < lm[3].x: count += 1
        else:
            if lm[4].x > lm[3].x: count += 1
        for tip in [8, 12, 16, 20]:
            if lm[tip].y < lm[tip - 2].y: count += 1
        return count

    def process(self, frame):
        # Preprocessing
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        hand_result = self.hands.process(rgb)
        face_result = self.face_mesh.process(rgb)

        # -----------------------------------------------------
        # 1. BLINK TOGGLE
        # -----------------------------------------------------
        if face_result.multi_face_landmarks:
            mesh = face_result.multi_face_landmarks[0].landmark
            ear_avg = (self._get_ear(mesh, self.LEFT_EYE) + 
                       self._get_ear(mesh, self.RIGHT_EYE)) / 2.0
            
            if ear_avg < self.EAR_THRESHOLD:
                self.blink_timestamps.append(time.time())
                
            now = time.time()
            self.blink_timestamps = [t for t in self.blink_timestamps if now - t < self.BLINK_WINDOW]
            
            if len(self.blink_timestamps) >= 3:
                self.active = not self.active
                self.blink_timestamps = []
                cv2.putText(frame, "SYSTEM TOGGLED", (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                time.sleep(0.5) 

        # -----------------------------------------------------
        # 2. HAND GESTURES
        # -----------------------------------------------------
        current_time = time.time()
        
        if self.active and hand_result.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Check Pinch First (Scrubbing logic)
                thumb = hand_landmarks.landmark[4]
                index = hand_landmarks.landmark[8]
                dist = math.hypot(thumb.x - index.x, thumb.y - index.y)
                
                if dist < self.PINCH_THRESHOLD:
                    # Pinch Logic - Instant response needed here for dragging
                    if self.last_pinch_x is not None:
                        delta = index.x - self.last_pinch_x
                        if abs(delta) > self.PINCH_MOVE_THRESHOLD:
                            # Faster cooldown for scrub actions (0.3s)
                            if current_time - self.last_action_time > 0.3:
                                if delta < 0: 
                                    pyautogui.press("right")
                                    cv2.putText(frame, "Seek FWD >>", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                                else:
                                    pyautogui.press("left")
                                    cv2.putText(frame, "<< Seek BWD", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                                self.last_action_time = current_time
                    self.last_pinch_x = index.x
                    # Reset non-pinch gesture timers
                    self.current_gesture = None
                
                else:
                    # Standard Gestures (Play, Vol, etc)
                    self.last_pinch_x = None
                    label = hand_result.multi_handedness[idx].classification[0].label
                    fingers = self._count_fingers(hand_landmarks, label)

                    # --- NEW STABILITY & HOLD LOGIC ---
                    
                    # 1. Is this a new gesture?
                    if fingers != self.current_gesture:
                        self.current_gesture = fingers
                        self.gesture_start_time = current_time # Start timer
                    
                    # 2. Calculate how long we've held it
                    hold_time = current_time - self.gesture_start_time
                    
                    # 3. Draw Loading Bar (Visual Feedback)
                    # This shows the user they need to hold the pose
                    if hold_time < self.HOLD_DURATION:
                        progress = int((hold_time / self.HOLD_DURATION) * 200)
                        cv2.rectangle(frame, (50, 140), (50 + progress, 150), (255, 255, 0), -1)
                        cv2.rectangle(frame, (50, 140), (250, 150), (255, 255, 255), 2)
                        cv2.putText(frame, f"Hold: {fingers}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                    
                    # 4. Trigger Action (Only if held long enough AND cooldown passed)
                    if hold_time >= self.HOLD_DURATION:
                         if current_time - self.last_action_time > self.POST_ACTION_COOLDOWN:
                            
                            action = None
                            if fingers == 5: action = "playpause"
                            elif fingers == 1: action = "volumeup"
                            elif fingers == 0: action = "volumedown"
                            elif fingers == 2: action = "nexttrack"
                            elif fingers == 3: action = "prevtrack"
                            
                            if action:
                                pyautogui.press(action)
                                # Visual Confirmation
                                cv2.putText(frame, f"ACTION: {action.upper()}", (50, 180), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                
                                # Reset Cooldown
                                self.last_action_time = current_time
                                # Reset Gesture so it doesn't auto-fire again instantly
                                self.gesture_start_time = current_time 

        # Status Overlay
        status_text = "ACTIVE" if self.active else "PAUSED"
        color = (0, 255, 0) if self.active else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame