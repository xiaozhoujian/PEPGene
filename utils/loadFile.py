#!/usr/bin/env python 
#-*- coding: UTF-8 -*- 
#=================================================================================================================

import os.path as path
from os.path import sep

#=================================================================================================================
## 给定相对路径，获取绝对路径
def get_abspath(this_file_path, *args):
	"""
	假设当前文件所在目录为A/B/C，目标文件是A/D/pep.txt
	调用load_file('..','..','D','pep.txt')即可，即相对路径
	"""
	return path.abspath(path.join(path.dirname(this_file_path), sep.join(args)))
	
#=================================================================================================================
def load_data(_abspath, _sep=','):
	"""加载测试数据
	文件每行默认代表一个pep，或一个mhc，或一个mhc对应的一组peps：
		单个序列时将矩阵折叠成一维向量
		多个序列时进行两次折叠
	返回结果为2维list
	"""
	dataset = []
	with open(_abspath, 'r') as f:
		for line in f:
			line = line.strip('\n')
			if len(line) != 0: 
				dataset.append([float(i) for i in line.split(_sep)])
	return dataset