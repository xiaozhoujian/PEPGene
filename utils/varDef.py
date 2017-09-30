#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================

import tensorflow as tf

def get_kernel_variable(shape, idx, regularizer):
	"""初始化3D核"""
	weights = tf.get_variable(
		"kernel_"+str(idx),
		shape,
		initializer=tf.truncated_normal_initializer(stddev=0.1)
	)
	tf.add_to_collection("losses", regularizer(weights))	
	return weights

def get_weight_variable(shape, regularizer=None, dtype=tf.float32):
	"""regularizer决定是否有正则化项，用于区分训练和测试"""
	weights = tf.get_variable(
		"weights",
		shape,
		dtype=dtype,
		initializer=tf.truncated_normal_initializer(stddev=0.1)
	)
	if regularizer != None:
		tf.add_to_collection("losses", regularizer(weights))	
	return weights
	
def get_bias_variable(shape, dtype=tf.float32):
	biases = tf.get_variable(
		"biases",
		shape,
		dtype=dtype,
		initializer=tf.constant_initializer(0.0)
	)
	return biases