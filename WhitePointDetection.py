import cv2
import numpy as np
import mediapipe as mp
# img_path = r'D:\pyProject\PalmDetection\dataset\5\Hand_0000187.jpg'
img_path = r'D:\pyProject\PalmDetection\dataset\WhitedotPalm.jpg'
img = cv2.imread(img_path)
cv2.namedWindow('WhitePointPalm',cv2.WINDOW_NORMAL)
cv2.imshow('WhitePointPalm',img)

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

imgHSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
cv2.namedWindow('imgHSV',cv2.WINDOW_NORMAL)
cv2.imshow('imgHSV',imgHSV)
# cv2.namedWindow('imgHSV0',cv2.WINDOW_NORMAL)
# cv2.namedWindow('imgHSV1',cv2.WINDOW_NORMAL)
# cv2.namedWindow('imgHSV2',cv2.WINDOW_NORMAL)
# cv2.imshow('imgHSV0',imgHSV[:,:,0])
# cv2.imshow('imgHSV1',imgHSV[:,:,1])
# cv2.imshow('imgHSV2',imgHSV[:,:,2])

low = np.array([0, 0, 221])
high = np.array([180, 30, 255])
dst = cv2.inRange(src=imgHSV, lowerb=low, upperb=high)  # HSV高低阈值，提取图像部分区域
# xy = np.column_stack(np.where(dst==255))
# for pt in xy:
#     xx, yy = pt[0], pt[1]
#     ps = (xx.astype(np.float32), yy.astype(np.float32))
    # 获取分割的区域，判断该点是否在区域里面
    # if :
    #     xx, yy = int(ps[0]), int(ps[1])
    #     cv2.circle(contour1, (xx,yy), 10, (0, 0, 255), 2)

cv2.namedWindow('dst',cv2.WINDOW_NORMAL)
cv2.imshow('dst', dst)
cv2.waitKey()