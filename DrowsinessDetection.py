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
DROWSY_COUNT_THRESH = 5    # how many drowsiness events before declaring driver unsafe

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

cap = cv2.VideoCapture(1)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,      # refines iris & lip landmarks (optional)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    ear_counter = 0
    mar_counter = 0
    alarm_on = False
    # cumulative drowsiness events counter and unsafe state
    drowsiness_count = 0
    unsafe_mode = False
    # button rect coords (x1, y1, x2, y2) will be updated per-frame
    button_coords = None

    # mouse callback to handle 'I'm ready' button clicks
    def on_mouse(event, x, y, flags, param):
        global unsafe_mode, drowsiness_count, button_coords
        if event == cv2.EVENT_LBUTTONDOWN and unsafe_mode and button_coords is not None:
            x1, y1, x2, y2 = button_coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                # user confirmed they are ready to drive again
                drowsiness_count = 0
                unsafe_mode = False
                print("User confirmed ready to drive. Counter reset.")

    # ensure window exists before setting mouse callback
    cv2.namedWindow("Drowsiness Detection")
    cv2.setMouseCallback("Drowsiness Detection", on_mouse)
    # variables to control beep timing (beep gets faster the longer drowsiness persists)
    last_beep_time = 0.0
    base_beep_interval = 1.0   # seconds between beeps when first detected
    min_beep_interval = 0.15   # fastest allowed interval
    unsafe_beep_interval = 0.4 # while unsafe_mode, keep beeping at this steady rate
    # fps calculation
    prev_time = time.time()
    fps = 0.0
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
        if not unsafe_mode:
            if ear > 0 and ear < EAR_THRESH:
                ear_counter += 1
            else:
                ear_counter = 0
                alarm_on = False

        # MAR logic
        if not unsafe_mode:
            if mar > MAR_THRESH:
                mar_counter += 1
            else:
                mar_counter = 0

        # Trigger drowsiness alarm if eyes closed for many frames
        if not unsafe_mode and ear_counter >= EAR_CONSEC_FRAMES:
            # on the frame where threshold is reached, count one drowsiness event
            if ear_counter == EAR_CONSEC_FRAMES:
                drowsiness_count += 1
            alarm_on = True
            cv2.putText(frame, "DROWSINESS ALERT - EYES CLOSED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # compute dynamic interval: reduce interval as ear_counter grows
            extra = max(0, ear_counter - EAR_CONSEC_FRAMES)
            interval = max(min_beep_interval, base_beep_interval - 0.01 * extra)
            now = time.time()
            if now - last_beep_time >= interval:
                beep()
                last_beep_time = now

        # Trigger yawn alert
        if not unsafe_mode and mar_counter >= MAR_CONSEC_FRAMES:
            # count one drowsiness event when threshold is reached
            if mar_counter == MAR_CONSEC_FRAMES:
                drowsiness_count += 1
            cv2.putText(frame, "YAWN ALERT", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            # compute dynamic interval based on mar_counter
            extra_m = max(0, mar_counter - MAR_CONSEC_FRAMES)
            interval_m = max(min_beep_interval, base_beep_interval - 0.01 * extra_m)
            now = time.time()
            if now - last_beep_time >= interval_m:
                beep()
                last_beep_time = now

        # compute FPS
        now_time = time.time()
        dt = now_time - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now_time

        # show metrics
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, h-40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # show cumulative drowsiness count
        cv2.putText(frame, f"Drowsiness Count: {drowsiness_count}", (w-280, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # show FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # If drowsiness count reaches threshold, mark unsafe and show a modal-like overlay
        if drowsiness_count >= DROWSY_COUNT_THRESH:
            unsafe_mode = True

        if unsafe_mode:
            # draw semi-transparent overlay
            overlay = frame.copy()
            alpha = 0.6
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # show warning text
            warn_text = "Driver has been deemed unsafe to drive"
            (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.putText(frame, warn_text, ((w - tw) // 2, h // 2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # draw confirmation button
            btn_w, btn_h = 340, 70
            bx = (w - btn_w) // 2
            by = h // 2 + 10
            cv2.rectangle(frame, (bx, by), (bx + btn_w, by + btn_h), (50, 205, 50), -1)
            btn_text = "I'm ready - Click to confirm"
            (btw, bth), _ = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, btn_text, (bx + (btn_w - btw) // 2, by + (btn_h + bth) // 2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # update button coords for mouse callback
            button_coords = (bx, by, bx + btn_w, by + btn_h)

            # keep beeping while unsafe until user confirms
            now = time.time()
            if now - last_beep_time >= unsafe_beep_interval:
                beep()
                last_beep_time = now

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
