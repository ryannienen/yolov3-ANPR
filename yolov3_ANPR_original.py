# test file if you want to quickly try tesseract on a license plate image
import pytesseract
import cv2
import numpy as np
import re
from datetime import datetime

# If you don't have tesseract executable in your PATH, include the following:
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

start_time = datetime.now()
# ================================================================================================
# 車牌辨識(START)

confidenceThreshold = 0.5
NMSThreshold = 0.3

modelConfiguration = 'cfg/yolov3.cfg'
modelWeights = 'yolov3_last.weights'

labelsPath = 'obj.names'
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

image = cv2.imread('images/car1.jpg')
# cv2.imwrite("car_plate_picture.jpg",image)
(H, W) = image.shape[:2]

#Determine output layer names
layerName = net.getLayerNames()
layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
net.setInput(blob)
layersOutputs = net.forward(layerName)

boxes = []
confidences = []
classIDs = []

# 蒐集符合confidenceThreshold的bounding box資訊
for output in layersOutputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > confidenceThreshold:
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY,  width, height) = box.astype('int')
            x = int(centerX - (width/2))
            y = int(centerY - (height/2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

# 車牌辨識(END)
# ================================================================================================

# ================================================================================================
# 車牌號碼個別辨識(START)

# # separate coordinates from box
# (xmin, ymin, xmax, ymax) = (boxes[0][0], boxes[0][1], boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3])
# box = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
# # colorful image
# # cv2.imshow("Colorful", box)
# # cv2.imwrite("car_plate_colorful.jpg",box)
# # grayscale region within bounding box
# gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
# # resize image to three times as large as original for better readability
# # gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
# # cv2.imshow("Gray", gray)
# # cv2.waitKey(0)
# # cv2.imwrite('Gray.jpg', gray)
# # perform gaussian blur to smoothen image
# blur = cv2.GaussianBlur(gray, (5,5), 0)
# # threshold the image using Otsus method to preprocess for tesseract
# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# # cv2.imshow("Otsu Threshold", thresh)
# # cv2.waitKey(0)
# # create rectangular kernel for dilation
# rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# # apply dilation to make regions more clear
# dilation = cv2.dilate(thresh, rect_kern, iterations = 1)

# # =================================================================
# # 增加erosion(START1)

# # cv2.imshow("Dilation", dilation)
# # cv2.imwrite('Dilation.jpg', dilation)
# # cv2.waitKey(0)
# erosion = cv2.erode(dilation, rect_kern, iterations = 1)
# # cv2.imshow("Erosion", erosion)
# # cv2.imwrite('Erosion.jpg', erosion)
# # cv2.waitKey(0)

# # 增加erosion(END1)
# # =================================================================

# # find contours of regions of interest within license plate
# try:
#     # =================================================================
#     # 增加erosion(START2)

#     # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # 增加erosion(END2)
#     # =================================================================

# except:
#     ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # sort contours left-to-right
# sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
# # create copy of gray image
# im2 = gray.copy()
# # create copy of colorful image
# im3 = box.copy()
# # create blank string to hold license plate number
# plate_num = ""
# # loop through contours and find individual letters and numbers in license plate
# for cnt in sorted_contours:
#     x,y,w,h = cv2.boundingRect(cnt)
#     height, width = im2.shape
#     # if height of box is not tall enough relative to total height then skip
#     if height / float(h) > 6:continue

#     # if height to width ratio is less than 1.5 skip
#     ratio = h / float(w)
#     if ratio < 1.5:continue

#     # if width is not wide enough relative to total width then skip
#     if width / float(w) > 15:continue

#     area = h * w
#     # if area is less than 100 pixels skip
#     if area < 100:continue

#     # draw the rectangle
#     rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
#     rect1 = cv2.rectangle(im3, (x,y), (x+w, y+h), (0,0,255),5)
#     # grab character region of image
#     roi = thresh[y-5:y+h+5, x-5:x+w+5]
#     # cv2.imshow("Character's orginal word", roi)
#     # perfrom bitwise not to flip image to black text on white background
#     roi = cv2.bitwise_not(roi)
#     # cv2.imshow("Character's bitwise_not word", roi)
#     # perform another blur on character region
#     roi = cv2.medianBlur(roi, 5)
#     # cv2.imshow("Character's single word", roi)
#     # cv2.waitKey(0)
#     try:
#         text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=012356789ABCDEFGHJKLMNPQRSTUVWXYZ --psm 7 --oem 3')
#         # clean tesseract text by removing any unwanted blank spaces
#         clean_text = re.sub('[\W_]+', '', text)
#         plate_num += clean_text
#     except: 
#         text = None
# if plate_num != None:
#     print("License Plate #: ", plate_num)
# # cv2.imshow("Character's Segmented gray", im2)
# # cv2.imwrite("car_plate_separate_gray.jpg", im2)
# cv2.imshow("Character's Segmented colorful", im3)
# # cv2.imwrite("car_plate_separate_colorful.jpg", im3)
# cv2.waitKey(0)

# 車牌號碼個別辨識(END)
# ================================================================================================

# ================================================================================================
# 直接辨識整張車牌(START)

(xmin, ymin, xmax, ymax) = (boxes[0][0], boxes[0][1], boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3])
box = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
# cv2.imshow("gray", gray)
# cv2.imwrite('car_plate.jpg', gray)
# gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
blur = cv2.GaussianBlur(gray, (5,5), 0)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
# cv2.imshow("thresh", thresh)
rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
# cv2.imshow("Dilation", dilation)
erosion = cv2.erode(dilation, rect_kern, iterations = 1)
# cv2.imshow("Erosion1", erosion)
roi = cv2.bitwise_not(erosion)
# cv2.imshow("Bitwise_not", roi)
roi = cv2.medianBlur(roi, 5)
cv2.imshow("MedianBlur", roi)
cv2.imwrite('car_plate_threshold.jpg', roi)
text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=012356789ABCDEFGHJKLMNPQRSTUVWXYZ --psm 7 --oem 3')
clean_text = re.sub('[\W_]+', '', text)
print("License Plate #: ", clean_text)
# cv2.imshow("License Plate", gray)
cv2.waitKey(0)

# 直接辨識整張車牌(END)
# ================================================================================================

# ================================================================================================
# 標出車牌信心值(START)

# Apply Non Maxima Suppression
detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

if(len(detectionNMS) > 0):
    for i in detectionNMS.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h),  (0, 0, 255), 10)
        text = '{}: {:.2f}'.format(labels[classIDs[i]], confidences[i])
        # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(image, text, (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)

cv2.imshow('Image', image)
# cv2.imwrite('car_plate_confidence.jpg', image)
cv2.waitKey(0)

# 標出車牌信心值(END)
# ================================================================================================
end_time = datetime.now()
during_time = end_time - start_time
print(f"--- 運行時間: {during_time} ---" )