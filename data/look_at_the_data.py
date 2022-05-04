import sys
import pandas as pd

names = ['label',
	'int_fea_00', 'int_fea_01', 'int_fea_02', 'int_fea_03', 'int_fea_04',
	'int_fea_05', 'int_fea_06', 'int_fea_07', 'int_fea_08', 'int_fea_09',
	'int_fea_10', 'int_fea_11', 'int_fea_12',
	'cat_fea_00', 'cat_fea_01', 'cat_fea_02', 'cat_fea_03', 'cat_fea_04',
	'cat_fea_05', 'cat_fea_06', 'cat_fea_07', 'cat_fea_08', 'cat_fea_09',
	'cat_fea_10', 'cat_fea_11', 'cat_fea_12', 'cat_fea_13', 'cat_fea_14',
	'cat_fea_15', 'cat_fea_16', 'cat_fea_17', 'cat_fea_18', 'cat_fea_19',
	'cat_fea_20', 'cat_fea_21', 'cat_fea_22', 'cat_fea_23', 'cat_fea_24',
	'cat_fea_25']

def load_data(file_path):
	return pd.read_csv(file_path, sep='\t', names=names)

if __name__ == '__main__':
	data = load_data(sys.argv[1])
	print('head lines:')
	print(data.head())
	print('info:')
	data.info()
	print('describe:')
	des = data.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	for key in des.keys():
		print(des[key])
	print('nunique:')
	print(data.nunique())
