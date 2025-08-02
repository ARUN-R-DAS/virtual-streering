import cv2
import mediapipe as mp
import math
import vgamepad as vg

#high means fast response
steering_sensitivity = 5
acc_sensitivity = 10
brake_sensitivity = 20
thresh_for_acc_brake = -0.015

threshold_foot_y = 315
apply_accelerator = False
apply_brake = False

left_foot_x, right_foot_x = None, None # initializing to avoid crash

# Initialize MediaPipe
mp_hands = mp.solutions.hands
# mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
# pose = mp_pose.Pose()

# Initialize virtual gamepad
gamepad = vg.VX360Gamepad()

hand_cam = cv2.VideoCapture(0)
foot_cam = cv2.VideoCapture(1)

# Turn detection (based on Y difference between hands)
def get_turn_direction(y_diff):
    if y_diff > -0.010:
        return "LEFT", abs(y_diff)
    elif y_diff < -0.05:
        return "RIGHT", abs(y_diff)
    else:
        return "STRAIGHT", 0.0

# Motion detection based on 2D hand distance
def get_motion_by_depth(z1,z2,thresh_for_acc_brake):
    avg_z = (z1 + z2) / 2
    diff = avg_z
    thresh_value = thresh_for_acc_brake
    if diff < thresh_value:
        return "ACCELERATE", diff, diff
    elif diff > thresh_value:
        return "BRAKE", abs(diff), diff
    else:
        return "IDLE", 0.0, diff

while True:
    ret, hand_frame = hand_cam.read()
    ret2, foot_frame = foot_cam.read()
    if not ret and not ret2:
        break

    hand_frame = cv2.flip(hand_frame, 1)
    foot_frame = cv2.flip(foot_frame, 1)
    hand_frame_rgb = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
    foot_frame_rgb = cv2.cvtColor(foot_frame, cv2.COLOR_BGR2RGB)
    hand_result = hands.process(hand_frame_rgb)
    # pose_result = pose.process(foot_frame_rgb)
    

    height, width, _ = hand_frame.shape

    if hand_result.multi_hand_landmarks and len(hand_result.multi_hand_landmarks) == 2:
        lm1 = hand_result.multi_hand_landmarks[0].landmark[9]  # palm centre
        lm2 = hand_result.multi_hand_landmarks[1].landmark[9]  

        # ----------------- BLACK PATCH BRAKE LOGIC ------------------
        # ---------------- TWO PATCHES: ACCELERATOR (left) + BRAKE (right) ----------------
        h, w, _ = foot_frame.shape
        roi_width = 100
        roi_height = 100
        y1 = h - roi_height - 20
        y2 = h - 20

        # Accelerator ROI (bottom-left)
        acc_x1 = 50
        acc_x2 = acc_x1 + roi_width
        acc_roi = foot_frame[y1:y2, acc_x1:acc_x2]

        # Brake ROI (bottom-right)
        brake_x2 = w - 50
        brake_x1 = brake_x2 - roi_width
        brake_roi = foot_frame[y1:y2, brake_x1:brake_x2]

        # Convert to grayscale and threshold (both ROIs)
        acc_gray = cv2.cvtColor(acc_roi, cv2.COLOR_BGR2GRAY)
        brake_gray = cv2.cvtColor(brake_roi, cv2.COLOR_BGR2GRAY)
        _, acc_mask = cv2.threshold(acc_gray, 50, 255, cv2.THRESH_BINARY_INV)
        _, brake_mask = cv2.threshold(brake_gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Calculate black pixel ratio
        acc_ratio = cv2.countNonZero(acc_mask) / (acc_mask.shape[0] * acc_mask.shape[1])
        brake_ratio = cv2.countNonZero(brake_mask) / (brake_mask.shape[0] * brake_mask.shape[1])

        # Apply controls if black ratio > 0.5
        apply_accelerator = acc_ratio > 0.5
        apply_brake = brake_ratio > 0.5

        # ---------------- Visualization ----------------
        cv2.rectangle(foot_frame, (acc_x1, y1), (acc_x2, y2), (0, 255, 0), 2)
        cv2.putText(foot_frame, f"ACC: {acc_ratio:.2f}", (acc_x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(foot_frame, (brake_x1, y1), (brake_x2, y2), (0, 0, 255), 2)
        cv2.putText(foot_frame, f"BRK: {brake_ratio:.2f}", (brake_x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #------------------hand stuff--------------------
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
        turn_val = max(min(y_diff * steering_sensitivity, 1.0), -1.0)  # 5 is sensitivity multiplier

        # Map acceleration/brake to 0-1 range
        throttle = 0.0
        brake = 0.0
        if apply_accelerator:
            throttle = .7
        elif apply_brake:
            brake = .7

        # Send input to virtual gamepad
        gamepad.left_joystick(x_value=int(turn_val * 32767), y_value=0) # 16 bit signed
        gamepad.right_trigger(value=int(throttle * 255))
        gamepad.left_trigger(value=int(brake * 255))
        gamepad.update()

        # Draw points and connecting line
        cv2.circle(hand_frame, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
        cv2.circle(hand_frame, (x2, y2), 8, (0, 255, 0), cv2.FILLED)
        cv2.line(hand_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Display steering info
        cv2.putText(hand_frame, f"Turn: {direction} y_diff = {y_diff}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(hand_frame, f"Turn Force: {turn_force:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Draw landmarks
    if hand_result.multi_hand_landmarks:
        for handLms in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(hand_frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Show the window
    cv2.imshow("Virtual Steering : hand_frame", hand_frame)
    cv2.imshow("Virtual Steering : foot_frame", foot_frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'),ord('Q')]:  # ESC to exit
        break

hand_cam.release()
foot_cam.release()
cv2.destroyAllWindows()
