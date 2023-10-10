import cv2 as cv
import numpy as np
Vision_Mode = True
lower_blue_contour = np.array([100, 43, 46])
upper_blue_contour = np.array([124, 255, 255])

lower_green_contour = np.array([40, 92, 46])
upper_green_contour = np.array([90, 255, 255])

lower_red_contour = np.array([160, 43, 46])
upper_red_contour = np.array([180, 255, 255])

def open_camera(num):
    cap = cv.VideoCapture(num)
    cap.set(cv.CAP_PROP_FPS, 30)
    return cap


def Color_Detect(Camera_num,color):
    cap = open_camera(Camera_num)
    max_perimeter_min = 800
    while True:
        _, img = cap.read()
        
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        if color is 'B':
            mask = cv.inRange(hsv, lower_blue_contour, upper_blue_contour)
        elif color is 'G':  
            mask = cv.inRange(hsv, lower_green_contour, upper_green_contour)
        elif color is 'R':
            mask = cv.inRange(hsv, lower_red_contour, upper_red_contour)
        if Vision_Mode:
            cv.imshow("mask",mask)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_perimeter = 0
        largest_contour = None

        # 遍历每个轮廓
        for cnt in contours:
            perimeter = cv.arcLength(cnt, True)
            if perimeter > max_perimeter:
                max_perimeter = perimeter
                largest_contour = cnt

        if largest_contour is not None and max_perimeter>max_perimeter_min:
            print(1)
        else:
            print(0)

        if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    if Vision_Mode:
        cv.destroyAllWindows()

