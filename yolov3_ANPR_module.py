import pytesseract
import cv2
import numpy as np
import re
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

def ANPR(image_url):

    # ================================================================================================
    # 車牌辨識(START)
    
    confidenceThreshold = 0.5
    
    modelConfiguration = 'cfg/yolov3.cfg'
    modelWeights = 'yolov3_last.weights'
    
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    
    image = cv2.imread(image_url)
    (H, W) = image.shape[:2]
    
    #Determine output layer names
    layerName = net.getLayerNames()
    layerName = [layerName[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB = True, crop = False)
    net.setInput(blob)
    layersOutputs = net.forward(layerName)
    
    boxes = []

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
    
    # 車牌辨識(END)
    # ================================================================================================
    
    # ================================================================================================
    # 辨識整張圖(START)
    
    (xmin, ymin, xmax, ymax) = (boxes[0][0], boxes[0][1], boxes[0][0]+boxes[0][2], boxes[0][1]+boxes[0][3])
    box = image[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
    gray = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
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
    # cv2.imshow("MedianBlur", roi)
    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=012356789ABCDEFGHJKLMNPQRSTUVWXYZ --psm 7 --oem 3')
    clean_text = re.sub('[\W_]+', '', text)
    print("License Plate #: ", clean_text)
    # cv2.imshow("License Plate", gray)
    # cv2.waitKey(0)
    
    # 辨識整張圖(END)
    # ================================================================================================
    return clean_text

if __name__ == "__main__":
    start_time = datetime.now()
    ANPR('images/car.jpg')
    end_time = datetime.now()
    during_time = end_time - start_time
    print(f"--- 運行時間: {during_time} ---" )