# -*- coding: utf-8 -*-
# !/usr/bin/python3
# YOLOv3_COCO，检测80种类别的物体

from ConstManager import *
import ModelManager
import hiai
from hiai.nn_tensor_lib import DataType
import numpy as np
import cv2
import utils
import datetime


'''
Yolo3_COCO模型, 输入H = 416, W = 416,模型的图像输入为RGB格式，这里使用OpenCV读取的图片，得到BGR格式的图像，在AIPP中完成BGR到RGB的色域转换和
image/255.0的归一化操作
bj_threshold 置信度阈值，取值范围为0~1。推理的时候，如果预测框的置信度小于该值，那么就会过滤掉, 默认为0.3
nms_threshold　NMS阈值，取值范围为0~1。默认为0.4
'''


class Yolo3_Resnet18Inference(object):
    def __init__(self):
        # 由用户指定推理引擎的所在Graph的id号
        self.graph_id = 1000
        self.model_engine_id = 100
        # 基于输入图片框坐标
        self.boxList = []
        # 置信度
        self.confList = []
        # 概率
        self.scoresList = []
        # 输入图片中行人部分
        self.personList = []
        # 实例化模型管理类
        self.model = ModelManager.ModelManager()
        self.width = 416
        self.height = 416
        # 描述推理模型以及初始化Graph
        self.graph = None
        self._getgraph()

    def __del__(self):
        self.graph.destroy()

    def _getgraph(self):
        # 描述推理模型
        inferenceModel = hiai.AIModelDescription('Yolo3_Resnet18', yolo3_resnet18_model_path)
        # 初始化Graph
        self.graph = self.model.CreateGraph(inferenceModel, self.graph_id, self.model_engine_id)
        if self.graph is None:
            print("Init Graph failed")

    '''
    1.定义输入Tensor的格式
    2.调用推理接口
    3.对一帧推理的正确结果保存到self.resultList中
    4.根据返回值True和False判断是否推理成功
    '''

    def Inference(self, input_image):
        if isinstance(input_image, np.ndarray) is None:
            return False

        # Image PreProcess
        resized_image = cv2.resize(input_image, (self.width, self.height))

        inputImageTensor = hiai.NNTensor(resized_image)
        nntensorList = hiai.NNTensorList(inputImageTensor)

        # 调用推理接口
        resultList = self.model.Inference(self.graph, nntensorList)

        if resultList is not None:
            bboxes = utils.get_result(resultList, self.width, self.height)  # 获取检测结果
            # print("bboxes:", bboxes)

            # Yolov_resnet18 Inference
            output_image = utils.draw_boxes(resized_image, bboxes)       # 在图像上画框
            output_image = cv2.resize(output_image, (input_image.shape[1], input_image.shape[0]))
            img_name = datetime.datetime.now().strftime("%Y-%m-%d%H-%M-%S-%f")
            cv2.imwrite('output_image/' + str(img_name) + '.jpg', output_image)

        else:
            print('no person in this frame.')
            return False

        return True


def dowork(src_img, yolo3_resnet18_app):

    res = yolo3_resnet18_app.Inference(src_img)
    if res is None:
        print("[ERROR] Please Check yolo3_resnet18_app.Inference!")
        return False
    else:
        # print("[ERROR] Run Failed, dowork function failed.")
        pass
    return True










