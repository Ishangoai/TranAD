import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float64,
                         delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
	temp = np.zeros(shape)
	with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
		ls = f.readlines()
	for line in ls:
		pos, values = line.split(':')[0], line.split(':')[1].split(',')
		start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
		temp[start-1:end-1, indx] = 1
	print(dataset, category, filename, temp.shape)
	np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
	a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
	return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = min(a), max(a)
	return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
	if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
	return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
	x = df[df.columns[3:]].values[::10, :]
	return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def load_data(dataset):
	folder = os.path.join(output_folder, dataset)
	os.makedirs(folder, exist_ok=True)
	if dataset == 'NAB':
		dataset_folder = 'data/NAB'
		file_list = os.listdir(dataset_folder)
		for filename in file_list:
			if not filename.endswith('.csv'): continue
			df = pd.read_csv(dataset_folder+'/'+filename)
			vals = df.values[:,1]
			min_temp, max_temp = np.min(vals), np.max(vals)
			vals = (vals - min_temp) / (max_temp - min_temp)
			train = vals.astype(float)
			train = train.reshape(-1, 1)
			fn = filename.replace('.csv', '')
			for file in ['train']:
				np.save(os.path.join(folder, f'{fn}_{file}.npy'), eval(file))
	elif dataset == 'SMD':
		dataset_folder = 'data/SMD'
		file_list = os.listdir(os.path.join(dataset_folder, "train"))
		for filename in file_list:
			if filename.endswith('.txt'):
				load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
				s = load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
				#load_and_save2('labels', filename, filename.strip('.txt'), dataset_folder, s)

if __name__ == '__main__':
	commands = sys.argv[1:]
	load = []
	if len(commands) > 0:
		for d in commands:
			load_data(d)
	else:
		print("Usage: python preprocess.py <datasets>")
		print(f"where <datasets> is space separated list of {datasets}")