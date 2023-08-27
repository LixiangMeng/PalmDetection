import cv2
import numpy as np
import mediapipe as mp
# img_path = r'D:\pyProject\PalmDetection\dataset\5\Hand_0000187.jpg'
img_path = r'D:\pyProject\PalmDetection\dataset\myhand.jpg'
img = cv2.imread(img_path)
# 图片镜像
img = cv2.flip(img, 1)
# cv2.imshow('palm image', img)
# img = cv2.rotate(img, cv2.ROTATE_180)
# 先通过掌纹识别手的正反面
#
# 再识别左右手

# mp.solutions.drawing_utils用于绘制
mp_drawing = mp.solutions.drawing_utils

# 参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

# mp.solutions.hands，是人的手
mp_hands = mp.solutions.hands

# 参数：1、是否检测静态图片，2、手的数量，3、检测阈值，4、跟踪阈值
hands_mode = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
image_hight, image_width, _ = img.shape
image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 处理RGB图像
results = hands_mode.process(image1)
print('Handedness:', results.multi_handedness)
if results.multi_handedness is None:
    print("不包含手")
else:
    print("包含手")
    for hand_landmarks in results.multi_hand_landmarks:
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
        )
        mp_drawing.draw_landmarks(
            img, hand_landmarks, mp_hands.HAND_CONNECTIONS, DrawingSpec_point, DrawingSpec_line)
img = cv2.flip(img, 1)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imwrite('image-hands.jpg', img)
hands_mode.close()
