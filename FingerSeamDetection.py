import cv2
import torch
import numpy as np
import mediapipe as mp
from skimage.draw import line


img_path = r'D:\pyProject\PalmDetection\dataset\VeinPalmRight.jpg'
img = cv2.imread(img_path)
cv2.namedWindow("palm image",cv2.WINDOW_NORMAL)
cv2.imshow("palm image", img)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) # 把图像转换到YUV色域
(y, cr, cb) = cv2.split(ycrcb) # 图像分割, 分别获取y, cr, br通道图像
# 高斯滤波, cr 是待滤波的源图像数据, (5,5)是值窗口大小, 0 是指根据窗口大小来计算高斯函数标准差
cr1 = cv2.GaussianBlur(cr, (5, 5), 0) # 对cr通道分量进行高斯滤波
# 根据OTSU算法求图像阈值, 对图像进行二值化
_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
res = cv2.bitwise_and(img,img, mask = skin)
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
dst = cv2.Laplacian(gray, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst)
h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #寻找轮廓
contour = h[0]
contour = sorted(contour, key = cv2.contourArea, reverse=True)#已轮廓区域面积进行排序
#contourmax = contour[0][:, 0, :]#保留区域面积最大的轮廓点坐标
bg = np.ones(dst.shape, np.uint8) *255#创建白色幕布
contour1 = cv2.drawContours(bg,contour[0],-1,(0,0,0),3) #绘制黑色轮廓


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
for hand_landmarks in results.multi_hand_landmarks:
    print('hand_landmarks:', hand_landmarks)
    print(
        f'Index finger tip coordinates: (',
        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
    )
for i in range(0,20):
    cv2.circle(contour1,
               (int(hand_landmarks.landmark[i].x * image_width), int(hand_landmarks.landmark[i].y * image_hight)), 10,
               (0, 0, 255), -1)



conts = np.transpose(contour[0],(1,0,2))
hull = cv2.convexHull(conts,returnPoints = False)
defects = cv2.convexityDefects(conts, hull)
if defects is not None:
    cnt = 0
    for i in range(defects.shape[0]): # calculate the angle
        s, e, f, d = defects[i][0]
        start = tuple(conts[0][s])
        end = tuple(conts[0][e])
        far = tuple(conts[0][f])
        a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # cosine theorem
        if angle <= np.pi / 2: # angle less than 90 degree, treat as fingers
            cnt += 1
            cv2.circle(contour1, far, 15, [0, 0, 255], -1)
    if cnt > 0:
        cnt = cnt+1
        cv2.putText(contour1, str(cnt), (0, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0) , 2, cv2.LINE_AA)

cv2.namedWindow("hull",cv2.WINDOW_NORMAL)
cv2.imshow("hull",contour1)
cv2.imwrite("2.jpg",contour1)
cv2.waitKey()
