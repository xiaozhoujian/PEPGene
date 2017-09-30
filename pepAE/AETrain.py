#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================

import tensorflow as tf
import common
import AEInference as aeInf
from utils import loadFile
from AEargs import *
tf_float_ = tf.float32 # 默认数据格式

#=================================================================================================================

## Forward propagation
def inference(onehot_pep, regularizer):
	"""当regularizer为None时返回编码结果，否则返回解码结果"""
	filters = [FILTER_HEIGHT_LIST, FILTER_DEPTH, FILTER_WIDTH]
	# Phase of encoding: outputs[BATCH_SIZE, KERNEL_TYPES*FILTER_DEPTH]
	outputs = aeInf.run_cnn_encoder(onehot_pep, filters, regularizer)
	if regularizer != None:
		# Phase of decoding: y_infer[BATCH_SIZE*15, 21]
		y_infer = aeInf.run_rnn_decoder(outputs, HIDDEN_SIZE, NUM_STEPS, regularizer)
		return y_infer
	return outputs
	
#=================================================================================================================

def train(inputs):
	## 输入和标签
	x = tf.placeholder(tf.int32, [None, NUM_STEPS * NUM_KEYS])
	y_real = tf.placeholder(tf.int32, [None, NUM_STEPS * NUM_KEYS])
	## 正则化
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
	## 输出
	y_infer = inference(x, regularizer)
	## 声明滑动平均对象
	global_step = tf.Variable(0, trainable = False)
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	## 定义损失函数
	y_label = tf.reshape(y_real, [-1, NUM_KEYS]) #(300,21)
	cross_entropy = tf.losses.softmax_cross_entropy(y_label, y_infer)
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE, 			#learning_rate
		global_step,					#global_step
		len(inputs)//BATCH_SIZE,    	#decay_steps
		LEARNING_RATE_DECAY				#decay_rate
	)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name = "train")
		
	## 初始化tf持久化类
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		num = len(inputs)
		for idx in range(TRAINING_STEPS):
			start_idx = idx * BATCH_SIZE % num
			end_idx = (idx+1) * BATCH_SIZE % num
			if end_idx > start_idx:
				xs = inputs[start_idx:end_idx]
			else:
				xs = inputs[start_idx:]+inputs[:end_idx]

			feed_dict = {x: xs, y_real: xs}
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = feed_dict)		
			#每ODEL_SAVE_CYC轮保存一次模型
			if idx%MODEL_SAVE_CYC == 0:
				print("After %d training step(s), loss on training batch is %g." % (idx, loss_value))
				saver.save(
					sess,
					path.join(MODEL_SAVE_PATH, MODEL_NAME),
					global_step = global_step
				)

#=================================================================================================================

def main(argv=None):
	input_data = loadFile.load_data(SRC_TRAIN_FILE)
	train(input_data)
	
if __name__ == '__main__':
	tf.app.run()