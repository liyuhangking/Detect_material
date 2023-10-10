import numpy as np
import cv2 as cv

Vision_Mode = True
TrackerBarMode = False
Camera_num = 0

# Initial HSV threshold values
lower_blue_contour = np.array([102, 69, 0])
upper_blue_contour = np.array([147, 255, 255])
lower_green_contour = np.array([38, 92, 46])
upper_green_contour = np.array([82, 255, 255])
lower_red_1_contour = np.array([0, 43, 46])
upper_red_1_contour = np.array([10, 255, 255])
lower_red_2_contour = np.array([156, 43, 46])
upper_red_2_contour = np.array([180, 255, 255])

lower_blue_circle = np.array([90, 0, 0])
upper_blue_circle = np.array([150, 255, 255])
lower_green_circle = np.array([38, 0, 46])
upper_green_circle = np.array([82, 255, 255])
lower_red_1_circle = np.array([0, 0, 46])
upper_red_1_circle = np.array([15, 255, 255])
lower_red_2_circle = np.array([156, 0, 46])
upper_red_2_circle = np.array([180, 255, 255])

thresh_blue = 60
thresh_green = 70
thresh_red = 64


if TrackerBarMode:
    # 初始化窗口和Trackbar
    cv.namedWindow("Contour Thresholds")
    cv.namedWindow("Circle Thresholds")
    cv.namedWindow("Gray Thresholds")
    # 创建回调函数，用于更新阈值变量
    
    def update_contour_thresholds(x):
        global lower_blue_contour, upper_blue_contour, lower_green_contour, upper_green_contour
        global lower_red_1_contour, upper_red_1_contour, lower_red_2_contour, upper_red_2_contour

        lower_blue_contour[0] = cv.getTrackbarPos("Lower Blue H", "Contour Thresholds")
        upper_blue_contour[0] = cv.getTrackbarPos("Upper Blue H", "Contour Thresholds")
        lower_green_contour[0] = cv.getTrackbarPos("Lower Green H", "Contour Thresholds")
        upper_green_contour[0] = cv.getTrackbarPos("Upper Green H", "Contour Thresholds")
        lower_red_1_contour[0] = cv.getTrackbarPos("Lower Red1 H", "Contour Thresholds")
        upper_red_1_contour[0] = cv.getTrackbarPos("Upper Red1 H", "Contour Thresholds")
        lower_red_2_contour[0] = cv.getTrackbarPos("Lower Red2 H", "Contour Thresholds")
        upper_red_2_contour[0] = cv.getTrackbarPos("Upper Red2 H", "Contour Thresholds")

        lower_blue_contour[1] = cv.getTrackbarPos("Lower Blue S", "Contour Thresholds")
        upper_blue_contour[1] = cv.getTrackbarPos("Upper Blue S", "Contour Thresholds")
        lower_green_contour[1] = cv.getTrackbarPos("Lower Green S", "Contour Thresholds")
        upper_green_contour[1] = cv.getTrackbarPos("Upper Green S", "Contour Thresholds")
        lower_red_1_contour[1] = cv.getTrackbarPos("Lower Red1 S", "Contour Thresholds")
        upper_red_1_contour[1] = cv.getTrackbarPos("Upper Red1 S", "Contour Thresholds")
        lower_red_2_contour[1] = cv.getTrackbarPos("Lower Red2 S", "Contour Thresholds")
        upper_red_2_contour[1] = cv.getTrackbarPos("Upper Red2 S", "Contour Thresholds")

        lower_blue_contour[2] = cv.getTrackbarPos("Lower Blue V", "Contour Thresholds")
        upper_blue_contour[2] = cv.getTrackbarPos("Upper Blue V", "Contour Thresholds")
        lower_green_contour[2] = cv.getTrackbarPos("Lower Green V", "Contour Thresholds")
        upper_green_contour[2] = cv.getTrackbarPos("Upper Green V", "Contour Thresholds")
        lower_red_1_contour[2] = cv.getTrackbarPos("Lower Red1 V", "Contour Thresholds")
        upper_red_1_contour[2] = cv.getTrackbarPos("Upper Red1 V", "Contour Thresholds")
        lower_red_2_contour[2] = cv.getTrackbarPos("Lower Red2 V", "Contour Thresholds")
        upper_red_2_contour[2] = cv.getTrackbarPos("Upper Red2 V", "Contour Thresholds")

    def update_circle_thresholds(x):
        global lower_blue_circle, upper_blue_circle, lower_green_circle, upper_green_circle
        global lower_red_1_circle, upper_red_1_circle, lower_red_2_circle, upper_red_2_circle

        lower_blue_circle[0] = cv.getTrackbarPos("Lower Blue H", "Circle Thresholds")
        upper_blue_circle[0] = cv.getTrackbarPos("Upper Blue H", "Circle Thresholds")
        lower_green_circle[0] = cv.getTrackbarPos("Lower Green H", "Circle Thresholds")
        upper_green_circle[0] = cv.getTrackbarPos("Upper Green H", "Circle Thresholds")
        lower_red_1_circle[0] = cv.getTrackbarPos("Lower Red1 H", "Circle Thresholds")
        upper_red_1_circle[0] = cv.getTrackbarPos("Upper Red1 H", "Circle Thresholds")
        lower_red_2_circle[0] = cv.getTrackbarPos("Lower Red2 H", "Circle Thresholds")
        upper_red_2_circle[0] = cv.getTrackbarPos("Upper Red2 H", "Circle Thresholds")

        lower_blue_circle[1] = cv.getTrackbarPos("Lower Blue S", "Circle Thresholds")
        upper_blue_circle[1] = cv.getTrackbarPos("Upper Blue S", "Circle Thresholds")
        lower_green_circle[1] = cv.getTrackbarPos("Lower Green S", "Circle Thresholds")
        upper_green_circle[1] = cv.getTrackbarPos("Upper Green S", "Circle Thresholds")
        lower_red_1_circle[1] = cv.getTrackbarPos("Lower Red1 S", "Circle Thresholds")
        upper_red_1_circle[1] = cv.getTrackbarPos("Upper Red1 S", "Circle Thresholds")
        lower_red_2_circle[1] = cv.getTrackbarPos("Lower Red2 S", "Circle Thresholds")
        upper_red_2_circle[1] = cv.getTrackbarPos("Upper Red2 S", "Circle Thresholds")

        lower_blue_circle[2] = cv.getTrackbarPos("Lower Blue V", "Circle Thresholds")
        upper_blue_circle[2] = cv.getTrackbarPos("Upper Blue V", "Circle Thresholds")
        lower_green_circle[2] = cv.getTrackbarPos("Lower Green V", "Circle Thresholds")
        upper_green_circle[2] = cv.getTrackbarPos("Upper Green V", "Circle Thresholds")
        lower_red_1_circle[2] = cv.getTrackbarPos("Lower Red1 V", "Circle Thresholds")
        upper_red_1_circle[2] = cv.getTrackbarPos("Upper Red1 V", "Circle Thresholds")
        lower_red_2_circle[2] = cv.getTrackbarPos("Lower Red2 V", "Circle Thresholds")
        upper_red_2_circle[2] = cv.getTrackbarPos("Upper Red2 V", "Circle Thresholds")

        # 创建回调函数，用于更新阈值变量
    def update_thresholds(x):
        global thresh_blue, thresh_green, thresh_red
        thresh_blue = cv.getTrackbarPos("Blue Threshold", "Gray Thresholds")
        thresh_green = cv.getTrackbarPos("Green Threshold", "Gray Thresholds")
        thresh_red = cv.getTrackbarPos("Red Threshold", "Gray Thresholds")

    # 创建Trackbar用于调整阈值
    cv.createTrackbar("Lower Blue H", "Contour Thresholds", lower_blue_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Blue H", "Contour Thresholds", upper_blue_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Blue S", "Contour Thresholds", lower_blue_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Blue S", "Contour Thresholds", upper_blue_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Blue V", "Contour Thresholds", lower_blue_contour[2], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Blue V", "Contour Thresholds", upper_blue_contour[2], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Green H", "Contour Thresholds", lower_green_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Green H", "Contour Thresholds", upper_green_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Green S", "Contour Thresholds", lower_green_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Green S", "Contour Thresholds", upper_green_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Green V", "Contour Thresholds", lower_green_contour[2], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Green V", "Contour Thresholds", upper_green_contour[2], 255, update_contour_thresholds)

    # 创建红色阈值的Trackbar
    cv.createTrackbar("Lower Red1 H", "Contour Thresholds", lower_red_1_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red1 H", "Contour Thresholds", upper_red_1_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Red1 S", "Contour Thresholds", lower_red_1_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red1 S", "Contour Thresholds", upper_red_1_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Red1 V", "Contour Thresholds", lower_red_1_contour[2], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red1 V", "Contour Thresholds", upper_red_1_contour[2], 255, update_contour_thresholds)

    cv.createTrackbar("Lower Red2 H", "Contour Thresholds", lower_red_2_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red2 H", "Contour Thresholds", upper_red_2_contour[0], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Red2 S", "Contour Thresholds", lower_red_2_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red2 S", "Contour Thresholds", upper_red_2_contour[1], 255, update_contour_thresholds)
    cv.createTrackbar("Lower Red2 V", "Contour Thresholds", lower_red_2_contour[2], 255, update_contour_thresholds)
    cv.createTrackbar("Upper Red2 V", "Contour Thresholds", upper_red_2_contour[2], 255, update_contour_thresholds)
    #----------------------------------------------------------------------------------------------------------------#
    # 创建Trackbar用于调整阈值
    cv.createTrackbar("Lower Blue H", "Circle Thresholds", lower_blue_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Blue H", "Circle Thresholds", upper_blue_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Blue S", "Circle Thresholds", lower_blue_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Blue S", "Circle Thresholds", upper_blue_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Blue V", "Circle Thresholds", lower_blue_circle[2], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Blue V", "Circle Thresholds", upper_blue_circle[2], 255, update_circle_thresholds)
    # 创建绿色阈值的Trackbar
    cv.createTrackbar("Lower Green H", "Circle Thresholds", lower_green_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Green H", "Circle Thresholds", upper_green_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Green S", "Circle Thresholds", lower_green_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Green S", "Circle Thresholds", upper_green_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Green V", "Circle Thresholds", lower_green_circle[2], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Green V", "Circle Thresholds", upper_green_circle[2], 255, update_circle_thresholds)

    # 创建红色阈值的Trackbar
    cv.createTrackbar("Lower Red1 H", "Circle Thresholds", lower_red_1_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red1 H", "Circle Thresholds", upper_red_1_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Red1 S", "Circle Thresholds", lower_red_1_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red1 S", "Circle Thresholds", upper_red_1_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Red1 V", "Circle Thresholds", lower_red_1_circle[2], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red1 V", "Circle Thresholds", upper_red_1_circle[2], 255, update_circle_thresholds)

    cv.createTrackbar("Lower Red2 H", "Circle Thresholds", lower_red_2_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red2 H", "Circle Thresholds", upper_red_2_circle[0], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Red2 S", "Circle Thresholds", lower_red_2_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red2 S", "Circle Thresholds", upper_red_2_circle[1], 255, update_circle_thresholds)
    cv.createTrackbar("Lower Red2 V", "Circle Thresholds", lower_red_2_circle[2], 255, update_circle_thresholds)
    cv.createTrackbar("Upper Red2 V", "Circle Thresholds", upper_red_2_circle[2], 255, update_circle_thresholds)
        # 创建Trackbar用于调整阈值
    cv.createTrackbar("Blue Threshold", "Gray Thresholds", thresh_blue, 255, update_thresholds)
    cv.createTrackbar("Green Threshold", "Gray Thresholds", thresh_green, 255, update_thresholds)
    cv.createTrackbar("Red Threshold", "Gray Thresholds", thresh_red, 255, update_thresholds)


def open_camera(num):
    cap = cv.VideoCapture(num)
    cap.set(cv.CAP_PROP_FPS, 30)
    return cap

def detect_and_draw_circles(img_path,color):
    # Create a window to display trackbars   
    img = img_path
    # 读取图像
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Initial HSV threshold values
    

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
    #cv.imshow('detected circles', res)
     # 将结果转换为灰度图像
    img_gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    
    img_gray = cv.medianBlur(img_gray, 5)
    #cv.imshow('detected circles', img_gray)q=
    if Vision_Mode:
        cimg = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT_ALT, 1, 20,
                              param1=50, param2=0.8, minRadius=30, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        all_centers = []

        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            all_centers.append(center)
            # draw the outer circle
            if Vision_Mode:
                cv.circle(img, center, circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            # cv.circle(cimg, center, 2, (0, 0, 255), 3)

        average_center = tuple(np.mean(all_centers, axis=0, dtype=np.int))
        x_offset = average_center[0] - center_x
        y_offset = average_center[1] - center_y

        horizontal_percentage = int((x_offset / (image_width / 2)) * 100)
        vertical_percentage = int(-(y_offset / (image_height / 2)) * 100)
        center = (horizontal_percentage,vertical_percentage)
        if Vision_Mode:
            cv.circle(img, average_center, 5, (255, 0, 0), -1)
        print("center " + color + ": " + str(center))
        if Vision_Mode:
            cv.imshow('detected circles', img)
    else:
        print("Not detected.")
        if Vision_Mode:
            cv.imshow('detected circles', img)

def contour_detect(img_path,color):
    img = img_path
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if color is 'B':
        mask = cv.inRange(hsv, lower_blue_contour, upper_blue_contour)
    elif color is 'G':  
        mask = cv.inRange(hsv, lower_green_contour, upper_green_contour)
    elif color is 'R':
        mask_red_1 = cv.inRange(hsv, lower_red_1_contour, upper_red_1_contour)
        mask_red_2 = cv.inRange(hsv, lower_red_2_contour, upper_red_2_contour)
        mask = cv.bitwise_or(mask_red_1,mask_red_2)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_perimeter = 0
    largest_contour =center= None

    # 遍历每个轮廓
    for cnt in contours:
        perimeter = cv.arcLength(cnt, True)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            largest_contour = cnt
    
    if largest_contour is not None:
        # 计算轮廓的中心坐标
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        epsilon = 0.05 * max_perimeter
        approx = cv.approxPolyDP(largest_contour, epsilon, True)
        # 绘制近似的轮廓
        if Vision_Mode:
            img = cv.drawContours(img, [approx], -1, (0, 0, 255), 4)
            cv.circle(img, center, 5, (0, 255, 0), -1)
        print("Center "+color+": "+str(center))
    else:
        print("Not Detected.")
    if Vision_Mode:
        cv.imshow('Contours', img)

# 流程：
# 提前设置灰度-二值化阈值
# 原图像转HSV->提取对应颜色->与原图像相与获得颜色->转灰度图->腐蚀->寻找轮廓->筛选最大面积的轮廓
# ->多边形拟合(再议，或可以直接求中心)->求中心
def contour_detect_v2_test(Camera_num,color):
    cap = open_camera(Camera_num)
    _, img = cap.read()

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    if color is 'B':
        mask = cv.inRange(hsv, lower_blue_contour, upper_blue_contour)
    elif color is 'G':  
        mask = cv.inRange(hsv, lower_green_contour, upper_green_contour)
    elif color is 'R':
        mask_red_1 = cv.inRange(hsv, lower_red_1_contour, upper_red_1_contour)
        mask_red_2 = cv.inRange(hsv, lower_red_2_contour, upper_red_2_contour)
        mask = cv.bitwise_or(mask_red_1,mask_red_2)

    res = cv.bitwise_and(img,img,mask = mask)
    if Vision_Mode:
        cv.imshow("inrange_res", res)

    imagegray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    if color is 'B':
        thresh = thresh_blue
    elif color is 'G':  
        thresh = thresh_green
    elif color is 'R':
        thresh = thresh_red


    _, imagethreshold = cv.threshold(imagegray, thresh, 255, cv.THRESH_BINARY)

    imagethreshold = cv.erode(imagethreshold, None, iterations=4)
    if Vision_Mode:
        cv.imshow("erode_res", imagethreshold)
    imagecontours, _ = cv.findContours(imagethreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    max_perimeter = 0
    largest_contour =center= None

    # 遍历每个轮廓
    for cnt in imagecontours:
        perimeter = cv.arcLength(cnt, True)
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            largest_contour = cnt
    if largest_contour is not None:
        M = cv.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            #center = (cx, cy)
            x_offset = cx - center_x
            y_offset = cy - center_y

            horizontal_percentage = int((x_offset / (image_width / 2)) * 100)
            vertical_percentage = int(-(y_offset / (image_height / 2)) * 100)
            center = (horizontal_percentage,vertical_percentage)
            if Vision_Mode:
                cv.circle(img, (cx,cy), 5, (0, 255, 0), -1)

        epsilon = 0.01 * cv.arcLength(largest_contour, True)
        approximations = cv.approxPolyDP(largest_contour, epsilon, True)

        if Vision_Mode:
            cv.drawContours(img, [approximations], 0, (0,0,255), 3)
            cv.imshow("Resulting_image", img)
        print("Center "+color+": "+str(center))
    else:
        print("Not Detected.")
        if Vision_Mode:
            cv.imshow("Resulting_image", img)




 
def contour_detect_v2_0(Camera_num,color):
    frame_count=0
    all_centers = []
    cap = open_camera(Camera_num)
    _, img = cap.read()
    while frame_count < 30:
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        if color is 'B':
            mask = cv.inRange(hsv, lower_blue_contour, upper_blue_contour)
        elif color is 'G':  
            mask = cv.inRange(hsv, lower_green_contour, upper_green_contour)
        elif color is 'R':
            mask_red_1 = cv.inRange(hsv, lower_red_1_contour, upper_red_1_contour)
            mask_red_2 = cv.inRange(hsv, lower_red_2_contour, upper_red_2_contour)
            mask = cv.bitwise_or(mask_red_1,mask_red_2)

        res = cv.bitwise_and(img,img,mask = mask)
        if Vision_Mode:
            cv.imshow("inrange_res", res)

        imagegray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

        if color is 'B':
            thresh = thresh_blue
        elif color is 'G':  
            thresh = thresh_green
        elif color is 'R':
            thresh = thresh_red


        _, imagethreshold = cv.threshold(imagegray, thresh, 255, cv.THRESH_BINARY)

        imagethreshold = cv.erode(imagethreshold, None, iterations=4)
        if Vision_Mode:
            cv.imshow("erode_res", imagethreshold)
        imagecontours, _ = cv.findContours(imagethreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        max_perimeter = 0
        largest_contour =center= None

        # 遍历每个轮廓
        for cnt in imagecontours:
            perimeter = cv.arcLength(cnt, True)
            if perimeter > max_perimeter:
                max_perimeter = perimeter
                largest_contour = cnt

        
        
        if largest_contour is not None:
            frame_count+=1
            print(1)
            M = cv.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                #center = (cx, cy)
                x_offset = cx - center_x
                y_offset = cy - center_y
                horizontal_percentage = int((x_offset / (image_width / 2)) * 100)
                vertical_percentage = int(-(y_offset / (image_height / 2)) * 100)

                center = (horizontal_percentage,vertical_percentage)

                all_centers.append(center)

                if Vision_Mode:
                    cv.circle(img, (cx,cy), 5, (0, 255, 0), -1)

            epsilon = 0.01 * cv.arcLength(largest_contour, True)
            approximations = cv.approxPolyDP(largest_contour, epsilon, True)

            if Vision_Mode:
                cv.drawContours(img, [approximations], 0, (0,0,255), 3)
                cv.imshow("Resulting_image", img)
            #print("Center "+color+": "+str(center))
        else:
            #print("Not Detected.")
            if Vision_Mode:
                cv.imshow("Resulting_image", img)
    cap.release()

    center = tuple(np.mean(all_centers, axis=0, dtype=np.int))
    print(center)
    return tuple(np.mean(all_centers, axis=0, dtype=np.int))

if __name__ == "__main__":    
    
    cap = cv.VideoCapture(Camera_num)
    cap.set(cv.CAP_PROP_FPS, 30)
    ret, img = cap.read()
    image_height, image_width, _ = img.shape
    center_x = image_width // 2
    center_y = image_height // 2
    cap.release()
    
    while True:
        # Capture frame-by-frame
        ret, img = cap.read()
        #detect_and_draw_circles(img,'G')
        contour_detect_v2_test(Camera_num,'B')
        #cv.imshow("lue",img)
        if Vision_Mode:
            # 检查是否按下 'q' 键来退出循环
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    # 释放摄像头资源和关闭窗口

    if Vision_Mode:
        cv.destroyAllWindows()
