import cv2
import mediapipe as mp
import math
import vgamepad as vg

threshold_foot_y = 315

# Initializing Mediapipe
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initializing Virtual gamepad
gamepad = vg.VX360Gamepad()

video = cv2.VideoCapture(1)

while video.isOpened():
    success, frame = video.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = pose.process(frame)

    # if result.pose_landmarks:
    #     mp_draw.draw_landmarks(
    #         frame,
    #         result.pose_landmarks,
    #         mp_pose.POSE_CONNECTIONS
    #     )
    if result.pose_landmarks:
        landmarks = result.pose_landmarks.landmark

        right_foot_tip = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        left_foot_tip = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
        
        #Get cordinates relative to image size
        h,w,_ = frame.shape
        left_x, left_y = int(left_foot_tip.x * w), int(left_foot_tip.y * h)
        right_x, right_y = int(right_foot_tip.x * w), int(right_foot_tip.y * h)

        #Drawing circles for visualization
        cv2.circle(frame, (left_x, left_y), 10, (255,0,0), -1)
        cv2.circle(frame, (right_x, right_y), 10, (0, 255, 0), -1)

        #pedals logic
        if left_y > threshold_foot_y:
            apply_brake = True
        else:
            apply_brake = False
        if right_y > threshold_foot_y:
            apply_accelerator = True
        else:
            apply_accelerator = False
        cv2.putText(
            frame,
            f"left_y : {str(left_y)} right_y : {str(right_y)}",
            (100,100),
            cv2.FONT_HERSHEY_DUPLEX,
            .8,
            (0,255,0),
            1
        )
        cv2.putText(
            frame,
            f"apply_brake : {apply_brake} apply_accelerator : {apply_accelerator}",
            (100,200),
            cv2.FONT_HERSHEY_DUPLEX,
            .5,
            (0,255,0),
            1
        )

    cv2.imshow("footcam feed",frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break
video.release()
cv2.destroyAllWindows()
