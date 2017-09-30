#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
import common
from utils import loadFile
from pepAE import AEargs

#=================================================================================================================

## 训练参数
TRAINING_STEPS = 3000
REGULARAZATION_RATE = 0.0001 #参数正则化比
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

## 模型保存和文件IO
MODEL_SAVE_CYC = 10
MODEL_SAVE_PATH = loadFile.get_abspath(__file__,'persisitance')
MODEL_NAME = "model.ckpt" 
SRC_FILE_TRAIN_MHC = loadFile.get_abspath(__file__,'..','data','mainData','train','mhc.txt')
SRC_FILE_TRAIN_PEP = loadFile.get_abspath(__file__,'..','data','mainData','train','pepsEigen.txt')
SRC_FILE_TRAIN_IC = loadFile.get_abspath(__file__,'..','data','mainData','train','ics.txt')
SRC_FILE_EVAL_MHC = loadFile.get_abspath(__file__,'..','data','mainData','eval','mhc.txt') 
SRC_FILE_EVAL_PEP = loadFile.get_abspath(__file__,'..','data','mainData','eval','pepsEigen.txt')
SRC_FILE_EVAL_IC = loadFile.get_abspath(__file__,'..','data','mainData','eval','ics.txt')

## 模型参数
BATCH_SIZE = 30 	# 训练时的batch大小，在使用时可能有差异 
NUM_PEPGROUP = 20				# 每对数据点中包含的pep数目
# MHC编码部分，二选一
	#if cnn
MHC_KERNEL_HEIGHT_LIST = [2,3,4,5,6]
MHC_FILTER_TYPES = len(MHC_KERNEL_HEIGHT_LIST)
MHC_FILTER_DEPTH = 10 
MHC_FILTER_WIDTH = 5
MHC_HEIGHT = MHC_FILTER_TYPES
MHC_WIDTH = MHC_FILTER_DEPTH
	#if rnn
MHC_HIDDEN_SIZE = 33 
MHC_HEIGHT = 10 
MHC_WIDTH = 10 
# pep特征矩阵
PEP_HEIGHT = AEargs.KERNEL_TYPES
PEP_WIDTH = AEargs.FILTER_DEPTH
# 突变表征矩阵R
MUTR_HEIGHT = 10
MUTR_WIDTH = 10
# 真实混合高斯分布参数
GMM_ARGS = []
