import cv2
import math
import vgamepad as vg
import numpy as np

steering_sensitivity = 1.0  # Not needed much here, since angle maps directly
acc_sensitivity = 10
brake_sensitivity = 20

# Gamepad
gamepad = vg.VX360Gamepad()

# Cameras
wheel_cam = cv2.VideoCapture(0)
foot_cam = cv2.VideoCapture(1)

# ROI setup (same)
roi_width, roi_height = 100, 100
acc_x1, acc_y1 = 50, 400
acc_x2, acc_y2 = acc_x1 + roi_width, acc_y1 + roi_height
brake_x1, brake_y1 = 500, 400
brake_x2, brake_y2 = brake_x1 + roi_width, brake_y1 + roi_height
selected_box, drag_offset = None, (0, 0)

acc_hold_duration = 0
brake_hold_duration = 0
throttle_play = 10

def mouse_callback(event, x, y, flags, param):
    global acc_x1, acc_y1, acc_x2, acc_y2
    global brake_x1, brake_y1, brake_x2, brake_y2, selected_box, drag_offset
    if event == cv2.EVENT_LBUTTONDOWN:
        if acc_x1 <= x <= acc_x2 and acc_y1 <= y <= acc_y2:
            selected_box = "acc"
            drag_offset = (x - acc_x1, y - acc_y1)
        elif brake_x1 <= x <= brake_x2 and brake_y1 <= y <= brake_y2:
            selected_box = "brake"
            drag_offset = (x - brake_x1, y - brake_y1)
    elif event == cv2.EVENT_MOUSEMOVE and selected_box:
        if selected_box == "acc":
            acc_x1 = x - drag_offset[0]
            acc_y1 = y - drag_offset[1]
            acc_x2 = acc_x1 + roi_width
            acc_y2 = acc_y1 + roi_height
        elif selected_box == "brake":
            brake_x1 = x - drag_offset[0]
            brake_y1 = y - drag_offset[1]
            brake_x2 = brake_x1 + roi_width
            brake_y2 = brake_y1 + roi_height
    elif event == cv2.EVENT_LBUTTONUP:
        selected_box = None

cv2.namedWindow("Virtual Steering : foot_frame")
cv2.setMouseCallback("Virtual Steering : foot_frame", mouse_callback)

while True:
    ret1, wheel_frame = wheel_cam.read()
    ret2, foot_frame = foot_cam.read()
    if not ret1 or not ret2:
        break

    wheel_frame = cv2.flip(wheel_frame, 1)
    foot_frame = cv2.flip(foot_frame, 1)

    # ---------- Red Tape Steering Logic ----------
    hsv = cv2.cvtColor(wheel_frame, cv2.COLOR_BGR2HSV)
    lower_red1 = (0, 100, 100)
    upper_red1 = (10, 255, 255)
    lower_red2 = (160, 100, 100)
    upper_red2 = (180, 255, 255)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    turn_val = 0.0  # Default

    if len(contours) >= 2:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        centers = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            centers.append((cx, cy))
            cv2.rectangle(wheel_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(wheel_frame, (cx, cy), 5, (255, 0, 0), -1)
        pt1, pt2 = centers
        cv2.line(wheel_frame, pt1, pt2, (0, 255, 255), 2)
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        turn_val = max(min(angle_deg / 90.0, 1.0), -1.0)  # Normalize to -1 to +1
        cv2.putText(wheel_frame, f"Angle: {angle_deg:.2f} deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(wheel_frame, f"Steering: {turn_val:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # ---------- Foot Logic (unchanged) ----------
    h, w, _ = foot_frame.shape
    acc_roi = foot_frame[acc_y1:acc_y2, acc_x1:acc_x2]
    brake_roi = foot_frame[brake_y1:brake_y2, brake_x1:brake_x2]
    acc_gray = cv2.cvtColor(acc_roi, cv2.COLOR_BGR2GRAY)
    brake_gray = cv2.cvtColor(brake_roi, cv2.COLOR_BGR2GRAY)
    _, acc_mask = cv2.threshold(acc_gray, 50, 255, cv2.THRESH_BINARY_INV)
    _, brake_mask = cv2.threshold(brake_gray, 50, 255, cv2.THRESH_BINARY_INV)
    acc_ratio = cv2.countNonZero(acc_mask) / (acc_mask.shape[0] * acc_mask.shape[1])
    brake_ratio = cv2.countNonZero(brake_mask) / (brake_mask.shape[0] * brake_mask.shape[1])
    apply_accelerator = acc_ratio > 0.5
    apply_brake = brake_ratio > 0.5

    if apply_accelerator:
        acc_hold_duration += 1
    else:
        acc_hold_duration = 0

    if apply_brake:
        brake_hold_duration += 1
    else:
        brake_hold_duration = 0

    throttle = min(1.0, acc_hold_duration / throttle_play)
    brake = min(1.0, brake_hold_duration / throttle_play)

    # ---------- Gamepad Output ----------
    gamepad.left_joystick(x_value=int(turn_val * 32767), y_value=0)
    gamepad.right_trigger(value=int(throttle * 255))
    gamepad.left_trigger(value=int(brake * 255))
    gamepad.update()

    # ---------- Display ----------
    cv2.rectangle(foot_frame, (acc_x1, acc_y1), (acc_x2, acc_y2), (0, 255, 0), 2)
    cv2.putText(foot_frame, f"ACC: {acc_ratio:.2f}", (acc_x1, acc_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.rectangle(foot_frame, (brake_x1, brake_y1), (brake_x2, brake_y2), (0, 0, 255), 2)
    cv2.putText(foot_frame, f"BRK: {brake_ratio:.2f}", (brake_x1, brake_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Virtual Steering : wheel_frame", wheel_frame)
    cv2.imshow("Virtual Steering : foot_frame", foot_frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

wheel_cam.release()
foot_cam.release()
cv2.destroyAllWindows()
