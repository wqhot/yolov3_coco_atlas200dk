# -*- coding: utf-8 -*-
# !/usr/bin/python3
# Author: Tianyi_Li
# Last Date: 2020/5/29
# YOLOv3_COCO，基于COCO数据集训练的TensorFlow版本，检测80种类别的物体。
# 用于创建Graph和Engine

import hiai
import time

class ModelManager(object):
    def __init__(self):
        pass

    '''
    初始化成功返回Graph实例,初始化失败返回None
    '''

    def CreateGraph(self, model,graph_id, model_engine_id):
        # 获取Graph实例
        myGraph = hiai.Graph(hiai.GraphConfig(graph_id = graph_id))
        if myGraph is None:
            print('get graph failed')
            return None
        with myGraph.as_default():
            model_engine = hiai.Engine(hiai.EngineConfig(engine_name='ModelInferenceEngine', side=hiai.HiaiPythonSide.Device,internal_so_name='/lib64/libhiai_python_device2.7.so',engine_id = model_engine_id))
            if model_engine is None:
                print('get model_engine failed')
                return None
            else:
                print('get model_engine ok!')
            with model_engine.as_default():
                if (None == model_engine.inference(input_tensor_list=hiai.NNTensorList(), ai_model=model)):
                    print('Init model_engine failed ')
                    return None
                else:
                    print('Init model_engine ok!')
        # 创建Graph
        if (hiai.HiaiPythonStatust.HIAI_PYTHON_OK == myGraph.create_graph()):
            print('create graph ok ')
            return myGraph
        else:
            print('create graph failed')
            return None

    '''
    传参失败或是推理失败,皆返回None
    '''

    def Inference(self, graphHandle, inputTensorList):
        if not isinstance(graphHandle, hiai.Graph):
            print("graphHandle is not Graph object")
            return None

        # 模型输入tensorlist
        resultList = graphHandle.proc(inputTensorList)
        if resultList is None:
            print('Inference error')
        return resultList
