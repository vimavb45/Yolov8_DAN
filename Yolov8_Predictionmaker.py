#!/usr/bin/python

import os
import sys
import matplotlib

from ultralytics import YOLO

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    #Data_path = 'C:\\Users\\vvebh\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\PhD\\Data\\NYUAD\\robot_data\\full_board\\test_1\\rgb\\'
    #Data_path = "C:/Users/vvebh/OneDrive - Danmarks Tekniske Universitet/Skrivebord/PhD/VS Code/YOLOv8_V4/Datasets/robot_data/full_board/test_1/rgb/"
    Data_path = "../YOLOv8_V4/Datasets/robot_data/full_guardrail/test_2/rgb/"
    #C:/Users/vvebh/OneDrive - Danmarks Tekniske Universitet/Skrivebord/PhD/Data/NYUAD/robot_data/full_guardrail/test_1/rgb
    Model_path = 'C:\\Users\\vvebh\\OneDrive - Danmarks Tekniske Universitet\\Skrivebord\\PhD\\VS Code\\YOLOv8\\runs\\segment\\train2\\weights\\best.pt'
    images = os.listdir(Data_path)
    #print(images)

    model = YOLO(Model_path)

    model.val(data = "Validation_Target_Dataset.yaml")

    #for image in images:
        #result = model(Data_path + '\\' + image)
        #print(result)
        #model.predict(Data_path + '\\' + image, save = True,imgsz = 640, conf = 0.8)

if __name__=="__main__":
    main()