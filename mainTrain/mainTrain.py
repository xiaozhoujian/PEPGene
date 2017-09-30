#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================
import tensorflow as tf
import common
from utils import varDef,loadFile
import mainInference as mainInf
from mainArgs import *

#=================================================================================================================

def train(input):
	
	##输入数据解析
	ds_mhc, ds_peps, ds_ics = input
	# 训练数据对总数
	num_samples = len(ds_mhc)
	
	## 输入和标签
	input_mhc = tf.placeholder(tf.int32, [None, None]) # 01 mat
	input_peps = tf.placeholder(mainInf.tf_float_, [None, NUM_PEPGROUP, PEP_HEIGHT*PEP_WIDTH])
	input_ics = tf.placeholder(mainInf.tf_float_, [None, NUM_PEPGROUP])
	
	## 参数正则化
	regularizer = tf.contrib.layers.l2_regularizer(REGULARAZATION_RATE)
	
	## 模型输出
	mutrs, output_peps, output_ics = mainInf.inference(input_mhc, input_peps, regularizer)

	## EMA
	global_step = tf.Variable(0, trainable=False)
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	
	## loss
	loss_normal = mainInf.calc_loss_of_distribution(mutrs, GMM_ARGS) #分布损失
	loss_peps = mainInf.calc_loss_of_peps(input_peps, output_peps) #生成peps损失
	loss_ics = mainInf.calc_loss_of_ics(input_ics, output_ics) #生成ics损失
	loss_reg = tf.add_n(tf.get_collection("losses")) #参数正则化损失
	loss = loss_normal + loss_peps + loss_ics + loss_reg
	
	## 学习率指数衰减
	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		num_samples//BATCH_SIZE,
		LEARNING_RATE_DECAY
	)
	
	##
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step = global_step)
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name = "train")
	
	##Save model
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for idx in range(TRAINING_STEPS):
			start_idx = idx*BATCH_SIZE%num_samples
			end_idx = (idx+1)*BATCH_SIZE%num_samples
			if end_idx > start_idx:
				batch_mhc = ds_mhc[start_idx:end_idx]
				batch_peps = ds_peps[start_idx:end_idx]
				batch_ics = ds_ics[start_idx:end_idx]
			else:
				batch_mhc = ds_mhc[start_idx:]+ds_mhc[:end_idx]
				batch_peps = ds_peps[start_idx:]+ds_peps[:end_idx]
				batch_ics = ds_ics[start_idx:]+ds_ics[:end_idx]

			feed_dict = {input_mhc: batch_mhc, input_peps: batch_peps, input_ics: batch_ics}
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = feed_dict)
			
			#每self.__MODEL_SAVE_CYC轮保存一次模型
			if idx%MODEL_SAVE_CYC == 0:
				print("After %d training step(s), loss on training batch is %g." % (idx, loss_value))
				saver.save(
					sess,
					path.join(MODEL_SAVE_PATH, MODEL_NAME),
					global_step = global_step
				)
				
#=================================================================================================================
def main(argv=None):
	input_mhc = loadFile.load_data(SRC_FILE_TRAIN_MHC)
	input_peps = loadFile.load_data(SRC_FILE_TRAIN_PEP)
	input_ics = loadFile.load_data(SRC_FILE_TRAIN_IC)
	train([input_mhc, input_peps, input_ics])

if __name__ == '__main__':
	tf.app.run()