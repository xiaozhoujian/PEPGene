#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================
import tensorflow as tf
import common
from utils import varDef

#=================================================================================================================
def run_cnn_encoder(inputs, filters, regularizer):
	"""
	args:
		inputs是一个稀疏矩阵，结构[BATCH_SIZE, 21*pep_length].此处可考虑固定pep_length=15
		filters = [filter_height_list, filter_depth, filter_width]
	return:
		固定维度的编码矩阵，结构[BATCH_SIZE, KERNEL_TYPES*FILTER_DEPTH]
	"""
	# tf.Variable对象声明使用方法get_weight_variable()和get_bias_variable()
	# 核的初始化使用get_kernel_variable()
	filter_height_list, filter_depth, filter_width = filters
	kernel_types = len(filter_height_list)
	pass
	return

#=================================================================================================================
## rnn decoder
def run_rnn_decoder(inputs, hidden_size, num_steps, regularizer):
	"""
	args:
		inputs: 来自inference_encoder的返回矩阵，结构[BATCH_SIZE, KERNEL_TYPES*FILTER_DEPTH]
		hidden_size: 状态维度
		num_steps: 循环次数
	return:
		和inference_encoder的输入矩阵相似，但结构为[BATCH_SIZE*15, 21]，便于计算交叉熵
	"""		
	#1 layer RNN
	with tf.variable_scope("rnn_decoder"):
		inputs = tf.stack([inputs]*num_steps, axis = 1) #(20,15,100)
		gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
		output, _ = tf.nn.dynamic_rnn(gru_cell, inputs, dtype=tf_float_) #(20,15,cell.output_size)
	#Full connection	
	with tf.variable_scope("fc_after_rnn"):
		output = tf.reshape(output, [-1, hidden_size])
		weights = varDef.get_weight_variable([hidden_size, hidden_size], regularizer)
		biases = varDef.get_bias_variable([hidden_size])
		output = tf.matmul(output, weights) + biases		
	return output