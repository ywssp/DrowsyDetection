# Save as drowsiness_mediapipe.py and run: python drowsiness_mediapipe.py
import cv2
import mediapipe as mp
import numpy as np
import time
import math
try:
    import winsound  # Windows beep fallback
    def beep(freq=1000, duration=200):
        winsound.Beep(freq, duration)
except Exception:
    # cross-platform fallback using OpenCV window flashing; you can replace with playsound or simpleaudio
    def beep(freq=1000, duration=200):
        print("\a", end="", flush=True)

# --- config / thresholds (tune these) ---
EAR_THRESH = 0.25          # EAR below this -> eye considered closed
EAR_CONSEC_FRAMES = 40     # number of consecutive frames with EAR < thresh to trigger drowsiness
MAR_THRESH = 0.6           # MAR above this -> possible yawn (tune)
MAR_CONSEC_FRAMES = 15     # consecutive frames to count as a yawn

# --- mediapipe indices we will use ---
# chosen 6 points per eye (P1..P6): left and right from many common implementations
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]   # P1..P6 (left)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]    # P1..P6 (right)
# mouth indices: use inner/top-bottom + corners (several options exist; this is robust enough)
MOUTH_TOP = 13   # inner upper lip
MOUTH_BOTTOM = 14 # inner lower lip
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# --- helpers ---
def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def compute_ear(landmarks, indices, img_w, img_h):
    # indices are [p1, p2, p3, p4, p5, p6] consistent with EAR formula
    pts = []
    for idx in indices:
        lm = landmarks[idx]
        pts.append((int(lm.x * img_w), int(lm.y * img_h)))
    # vertical distances
    A = euclidean(pts[1], pts[5])  # p2 - p6
    B = euclidean(pts[2], pts[4])  # p3 - p5
    C = euclidean(pts[0], pts[3])  # p1 - p4 (horizontal)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def compute_mar(landmarks, img_w, img_h):
    top = landmarks[MOUTH_TOP]
    bottom = landmarks[MOUTH_BOTTOM]
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top_pt = (int(top.x * img_w), int(top.y * img_h))
    bottom_pt = (int(bottom.x * img_w), int(bottom.y * img_h))
    left_pt = (int(left.x * img_w), int(left.y * img_h))
    right_pt = (int(right.x * img_w), int(right.y * img_h))
    vertical = euclidean(top_pt, bottom_pt)
    horizontal = euclidean(left_pt, right_pt)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

# --- initialize mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # refines iris & lip landmarks (optional)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    ear_counter = 0
    mar_counter = 0
    alarm_on = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        ear = 0.0
        mar = 0.0
        if results.multi_face_landmarks:
            # take first face only
            landmarks = results.multi_face_landmarks[0].landmark

            # compute left & right EAR
            left_ear = compute_ear(landmarks, LEFT_EYE_IDX, w, h)
            right_ear = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
            ear = (left_ear + right_ear) / 2.0

            # compute MAR
            mar = compute_mar(landmarks, w, h)

            # draw some landmarks (optional)
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0],
                                      mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None,
                                      connection_drawing_spec=mp_drawing.DrawingSpec((0,255,0), 1, 1))

        # EAR logic
        if ear > 0 and ear < EAR_THRESH:
            ear_counter += 1
        else:
            ear_counter = 0
            alarm_on = False

        # MAR logic
        if mar > MAR_THRESH:
            mar_counter += 1
        else:
            mar_counter = 0

        # Trigger drowsiness alarm if eyes closed for many frames
        if ear_counter >= EAR_CONSEC_FRAMES:
            alarm_on = True
            cv2.putText(frame, "DROWSINESS ALERT - EYES CLOSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            beep()

        # Trigger yawn alert
        if mar_counter >= MAR_CONSEC_FRAMES:
            cv2.putText(frame, "YAWN ALERT", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            beep()

        # show metrics
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
