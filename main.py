# -*- coding: utf-8 -*-
# !/usr/bin/python3
# Author: Tianyi_Li
# Last Date: 2020/5/29
# YOLOv3_COCO，基于COCO数据集训练的TensorFlow版本，检测80种类别的物体。
# 主函数部分

import sys
import re
import cv2
import yolo3_resnet18_inference
import time
import datetime


# Get Video
lenofUrl = len(sys.argv)

# The number of parameters is incorrect.
if lenofUrl <= 1:
    print("[ERROR] Please input mp4/Rtsp URL")
    sys.exit()
elif lenofUrl >= 3:
    print("[ERROR] param input Error")
    sys.exit()

URL = sys.argv[1]

# match Input parameter format
URL1 = re.match('rtsp://', URL)
URL2 = re.search('.mp4', URL)


# Determine if it is a mp4 video based on matching rules
if URL1 is None:
    if URL2 is None:
        print("[ERROR] should input correct URL")
        sys.exit()
    else:
        mp4_url = True
else:
    mp4_url = False

# Init Graph and Engine
yolo3_resnet18_app = yolo3_resnet18_inference.Yolo3_Resnet18Inference()
if yolo3_resnet18_app.graph is None:
        sys.exit(1)

# Get Start time
run_starttime = datetime.datetime.now()

# Get Frame
cap = cv2.VideoCapture(URL)
ret, frame = cap.read()
print("视频是否打开成功:", ret)

# Get Video Information
frames_num = cap.get(7)
frame_width = cap.get(3)
frame_height = cap.get(4)

# According to the flag,Perform different processing methods
if mp4_url:
    try:
        while ret:
            # Processing the detection results of a frame of pictures
            strattime = time.time()
            ret = yolo3_resnet18_inference.dowork(frame, yolo3_resnet18_app)
            endtime = time.time()
            print('Process this image cost time: ' + str((endtime - strattime) * 1000) + 'ms')
            if ret is None:
                sys.exit(1)

            # Loop through local video files
            ret, frame = cap.read()

        # Run done, print input video information
        run_endtime = datetime.datetime.now()
        run_time = (run_endtime - run_starttime).seconds
        print("输入视频的宽度:", frame_width)
        print("输入视频的高度:", frame_height)
        print("输入视频的总帧数:", frames_num)
        print("程序运行总时间:" + str(run_time) + "s")
        print('-------------------------end')
    except Exception as e:
        print("ERROR", e)
    finally:
        # Turn off the camera
        cap.release()
else:
    print("[ERROR] Run Failed, please check input video.")


