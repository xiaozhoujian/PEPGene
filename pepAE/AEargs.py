#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
from utils import loadFile

#=================================================================================================================

## constant

NUM_KEYS = 22 # 编码字典键数

#=================================================================================================================

## file IO

MODEL_SAVE_CYC = 10 
MODEL_SAVE_PATH = loadFile.get_abspath(__file__,'persistance') #模型保存路径
MODEL_NAME = "model.ckpt" # 模型保存文件名

SRC_TRAIN_FILE = loadFile.get_abspath(__file__, '..', 'data', 'aeData', 'train.txt') # 训练时的输入文件
SRC_EVAL_FILE = loadFile.get_abspath(__file__, '..', 'data', 'aeData','eval.txt') # 测试时的输入文件

# 全编码文件IO
SRC_FILE_EVAL = loadFile.get_abspath(__file__,'..','data','mainData','eval','peps.txt')
SRC_FILE_TRAIN = loadFile.get_abspath(__file__,'..','data','mainData','train','peps.txt')
OUT_FILE_EVAL = loadFile.get_abspath(__file__,'..','data','mainData','eval','pepsEigen.txt')
OUT_FILE_TRAIN = loadFile.get_abspath(__file__,'..','data','mainData','train','pepsEigen.txt')

## 模型参数

BATCH_SIZE = 20

# encoder: mng-CNN
FILTER_HEIGHT_LIST = [2,3,4,5,6] #核height取值list
FILTER_DEPTH = 20 #每类核的深度
FILTER_WIDTH = 21 #所有核的宽度，定值
KERNEL_TYPES = len(FILTER_HEIGHT_LIST) #核height的总类数
FILTERS = [FILTER_HEIGHT_LIST, FILTER_DEPTH, FILTER_WIDTH]

# decoder: 1 layer GRU
HIDDEN_SIZE = 21
NUM_STEPS = 15 #RNN循环次数，15是pep最大长度

## 训练参数

TRAINING_STEPS = 3000
REGULARAZATION_RATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

## 测试时模型加载间隔时间
EVAL_INTERVAL_SECS = 10