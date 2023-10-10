import cv2 as cv
import numpy as np
Vision_Mode = True

lower_blue_circle = np.array([90, 0, 0])
upper_blue_circle = np.array([150, 255, 255])
lower_green_circle = np.array([38, 0, 46])
upper_green_circle = np.array([82, 255, 255])
lower_red_1_circle = np.array([0, 0, 46])
upper_red_1_circle = np.array([15, 255, 255])
lower_red_2_circle = np.array([156, 0, 46])
upper_red_2_circle = np.array([180, 255, 255])

def open_camera(num):
    cap = cv.VideoCapture(num)
    cap.set(cv.CAP_PROP_FPS, 30)
    return cap

def detect_and_draw_circles(Camera_num,color):
    cap = open_camera(Camera_num)
    frame_count = 0
    if Vision_Mode:
        average_center_vision = (0,0)
    while True:
        _, img = cap.read()

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        if color is 'B':
            mask_blue = cv.inRange(hsv, lower_blue_circle, upper_blue_circle)
            res = cv.bitwise_and(img, img, mask=mask_blue)
        elif color is 'G':
            mask_green = cv.inRange(hsv, lower_green_circle, upper_green_circle)
            res = cv.bitwise_and(img, img, mask=mask_green)
        elif color is 'R':
            mask_red_1 = cv.inRange(hsv, lower_red_1_circle, upper_red_1_circle)
            mask_red_2 = cv.inRange(hsv, lower_red_2_circle, upper_red_2_circle)
            mask_red = cv.bitwise_or(mask_red_1,mask_red_2)
            res = cv.bitwise_and(img, img, mask=mask_red)
        # 将结果转换为灰度图像
        img_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        
        img_gray = cv.medianBlur(img_gray, 5)

        circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT_ALT, 1, 20,
                                param1=50, param2=0.8, minRadius=30, maxRadius=0)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            all_centers = []

            for circle in circles[0, :]:
                center = (circle[0], circle[1])
                all_centers.append(center)

                if Vision_Mode:
                    cv.circle(img, center, circle[2], (0, 255, 0), 2)
            frame_count+=1
            if(frame_count>15):
                average_center = tuple(np.mean(all_centers, axis=0, dtype=np.int))
                if Vision_Mode:
                    average_center_vision = average_center
                print(average_center[0],average_center[1])
                frame_count = 0
                all_centers = []

            if Vision_Mode:
                if average_center_vision !=(0,0):
                    cv.circle(img, average_center, 5, (255, 0, 0), -1)
                
                cv.imshow('detected circles', img)
        else:
            print(0,0)
            if Vision_Mode:
                cv.imshow('detected circles', img)
        if cv.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
