#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================

import time
import tensorflow as tf

import common
from utils import varDef, loadFile
from AETrain import tf_float_, inference
import AEInference as aeInf 
from mainTrain import mainInference as mainInf
from AEargs import \
	FILTERS,\
	NUM_STEPS,\
	NUM_KEYS,\
	MOVING_AVERAGE_DECAY,\
	MODEL_SAVE_PATH,\
	EVAL_INTERVAL_SECS,\
	SRC_EVAL_FILE
	

#=================================================================================================================
def test(input_data):

	with tf.Graph().as_default() as g:	
		## 输入输出
		x = tf.placeholder(tf_float_, [None, NUM_STEPS * NUM_KEYS], name="input_x") # 315 = 15 * 21
		y_real = tf.placeholder(tf_float_, [None, NUM_STEPS * NUM_KEYS], name="input_y_real") # y_real == x
		#input_pep = loadFile.load_data(loadFile.get_abspath(file_name))
		testing_dict = {x: input_data, y_real: input_data}
		
		## 计算前向结果并和真实值比较计算距离
		outputs = aeInf.run_cnn_encoder(x, FILTERS, None)
		y_infer = aeInf.run_rnn_decoder(outputs, HIDDEN_SIZE, NUM_STEPS, None)
		y_real = tf.reshape(y_real, [-1, NUM_KEYS])
		distance = mainInf.calc_loss_of_peps(y_real, y_infer)
	
		## 通过变量重命名的方式加载模型
		variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)
		
		## 测试：每隔？秒计算一次损失
		while True:
			with tf.Session() as sess:
				## 查找最新模型
				ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
				##
				if ckpt and ckpt.model_checkpoint_path:
					## 加载模型
					saver.restore(sess, ckpt.model_checkpoint_path)
					## 求模型的迭代轮数
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					distance_value = sess.run(distance, feed_dict=testing_dict)
					print("After %s training step(s), distance = %g"%(global_step, distance_value))
				else:
					print("No checkpoint file found")
					return
				time.sleep(EVAL_INTERVAL_SECS)

#=================================================================================================================				
def main(argv=None):
	input_pep = loadFile.load_data(SRC_EVAL_FILE)
	test(input_pep)

if __name__ == '__main__':
	tf.app.run()