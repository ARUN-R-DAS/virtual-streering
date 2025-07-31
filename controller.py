import cv2
import mediapipe as mp
import math
import vgamepad as vg

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

# Initialize virtual gamepad
gamepad = vg.VX360Gamepad()

cap = cv2.VideoCapture(0)

# Turn detection (based on Y difference between hands)
def get_turn_direction(y_diff):
    if y_diff > 0.05:
        return "LEFT", abs(y_diff)
    elif y_diff < -0.05:
        return "RIGHT", abs(y_diff)
    else:
        return "STRAIGHT", 0.0

# Motion detection based on 2D hand distance
def get_motion_2d(x1, y1, x2, y2, neutral_dist=200):
    dist = math.hypot(x2 - x1, y2 - y1)
    diff = dist - neutral_dist
    if diff > 150:
        return "ACCELERATE", diff, diff
    elif diff < 100:
        return "BRAKE", abs(diff), diff
    else:
        return "IDLE", 0.0, diff

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    height, width, _ = frame.shape

    if result.multi_hand_landmarks and len(result.multi_hand_landmarks) == 2:
        lm1 = result.multi_hand_landmarks[0].landmark[0]  # Wrist 1
        lm2 = result.multi_hand_landmarks[1].landmark[0]  # Wrist 2

        x1, y1 = int(lm1.x * width), int(lm1.y * height)
        x2, y2 = int(lm2.x * width), int(lm2.y * height)

        # Identify left/right by x
        if x1 < x2:
            left_y, right_y = y1, y2
        else:
            left_y, right_y = y2, y1

        # Calculate turn
        y_diff = (right_y - left_y) / height
        direction, turn_force = get_turn_direction(y_diff)

        # Normalize turn to range -1.0 to 1.0
        turn_val = max(min(y_diff * 5, 1.0), -1.0)  # 5 is sensitivity multiplier

        # Calculate 2D distance for acceleration/braking
        motion, accel_force, dist_diff = get_motion_2d(x1, y1, x2, y2)

        # Map acceleration/brake to 0-1 range
        throttle = 0.0
        brake = 0.0
        if motion == "ACCELERATE":
            throttle = min(accel_force / 300.0, 1.0)
        elif motion == "BRAKE":
            brake = min(accel_force / 300.0, 1.0)

        # Send input to virtual gamepad
        gamepad.left_joystick(x_value=int(turn_val * 32767), y_value=0)
        gamepad.right_trigger(value=int(throttle * 255))
        gamepad.left_trigger(value=int(brake * 255))
        gamepad.update()

        # Draw points and connecting line
        cv2.circle(frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Display steering info
        cv2.putText(frame, f"Turn: {direction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Turn Force: {turn_force:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display acceleration/brake info
        cv2.putText(frame, f"Motion: {motion}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        cv2.putText(frame, f"Force: {accel_force:.2f}", (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show 2D distance difference
        cv2.putText(frame, f"Dist diff: {dist_diff:.1f}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    # Draw landmarks
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Show the window
    cv2.imshow("Virtual Steering Joystick", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
