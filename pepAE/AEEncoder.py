#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================
import tensorflow as tf
import common
from utils import varDef, loadFile
from AETrain import tf_float_, inference
import AEInference as aeInf
from AEargs import \
	NUM_STEPS,\
	NUM_KEYS,\
	MODEL_SAVE_PATH,\
	SRC_FILE_EVAL,\
	SRC_FILE_TRAIN,\
	OUT_FILE_EVAL,\
	OUT_FILE_TRAIN
# 待编码文件每行有多少个peps
 from mainTrain.mainArgs import NUM_PEPGROUP 
 
#=================================================================================================================
def encodePEP(input_file, output_file):
	with tf.Graph().as_default() as g:
		x = tf.placeholder(tf_float_, [None, NUM_STEPS * NUM_KEYS], name="input_x")
		outputs = inference(x, None)
		
		saver = tf.train.Saver() #导入部分参数？
		with tf.Session() as sess:
			## 查找最新模型
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				
				with open(output_file,'w') as fout:
					pep_length = NUM_STEPS*NUM_KEYS
					for line in loadFile.load_data(input_file):
						# line是NUM_PEPGROUP个pep经过onehot组成的
						batch_pep_oh = []
						if(pep_length != len(line) / NUM_PEPGROUP):
							print("Invalid line length in input file!")
							return
						for i in range(NUM_PEPGROUP):
							batch_pep_oh.append(line[i*pep_length:(i+1)*pep_length])
						#[NUM_PEPGROUP, PEP特征矩阵大小]
						result = sess.run(outputs, feed_dict={x: batch_pep_oh})
						# 取决于result的数据类型？
						result = tf.reshape(result, [-1])
						fout.write(','.join([str(n) for n in result]))
				
			else:
				print("No checkpoint file found")
				return

#=================================================================================================================				

encodePEP(SRC_FILE_EVAL, OUT_FILE_EVAL)
encodePEP(SRC_FILE_TRAIN, OUT_FILE_TRAIN)