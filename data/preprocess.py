import sys
import json
import math
import argparse
from collections import defaultdict

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

class Preprocesser:

	def __init__(self, filename=None):
		if filename == None:
			self.sta_dict = {}
		else:
			with open(filename, 'r') as fin:
				self.sta_dict = json.load(fin)

	def fit(self, filename):
		fin = open(filename, 'r')
		for line in fin:
			content = line.rstrip('\n').split('\t')
			if len(content) != len(names):
				continue
			for i in range(len(content)):
				k = names[i]
				v = content[i]
				if v == '':
					continue
				if k.startswith('int_fea'):
					v = float(v)
					if k not in self.sta_dict:
						self.sta_dict[k] = {'min': v, 'max': v}
					else:
						self.sta_dict[k]['min'] = min(self.sta_dict[k]['min'], v)
						self.sta_dict[k]['max'] = max(self.sta_dict[k]['max'], v)
				elif k.startswith('cat_fea'):
					if k not in self.sta_dict:
						self.sta_dict[k] = defaultdict(int)
					self.sta_dict[k][v] += 1
		fin.close()

	def _log_normalization(self, v, mn, mx):
		if v < mn:
			return 0
		if v > mx:
			return 1.0
		return round(math.log(v - mn + 1, 10) / math.log(mx - mn + 1, 10), 4)

	def _uniform_scale(self, v, scale):
		# v是[0, 1]浮点数
		return int(v * scale)

	def transform(self, infilename, outfilename, cat_min_num=10, str2idx=False):
		fin = open(infilename, 'r')
		fout = open(outfilename, 'w')
		cat2idx_dict = {}
		if str2idx:
			for key, val in self.sta_dict.items():
				if key.startswith('cat_fea'):
					if key not in cat2idx_dict:
						cat2idx_dict[key] = {}
						cat2idx_dict[key][''] = 0
					for cat_v, num in val.items():
						if cat_v not in cat2idx_dict[key] and num >= cat_min_num:
							cat2idx_dict[key][cat_v] = len(cat2idx_dict[key])
		for line in fin:
			content = line.rstrip('\n').split('\t')
			if len(content) != len(names):
				continue
			new_line = []
			for i in range(len(content)):
				k = names[i]
				v = content[i]
				if k.startswith('int_fea'):
					trans_v = '0' if v == '' else self._uniform_scale(self._log_normalization(float(v), self.sta_dict[k]['min'], self.sta_dict[k]['max']), 1000)
				elif k.startswith('cat_fea'):
					if str2idx:
						trans_v = '0' if v not in cat2idx_dict[k] else cat2idx_dict[k][v]
					else:
						trans_v = '' if v not in self.sta_dict[k] or self.sta_dict[k][v] < cat_min_num else v
				else:
					trans_v = v
				new_line.append(trans_v)
			print('\t'.join(map(str, new_line)), file=fout)
		fout.close()
		fin.close()

	def dump_sta_dict(self, filename):
		with open(filename, 'w') as fout:
			json.dump(self.sta_dict, fout)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('infilename', help='输入文件')
	parser.add_argument('stafilename', help='sta文件')
	parser.add_argument('action', choices=['fit', 'transform'])
	parser.add_argument('--outfilename', help='输出文件')
	parser.add_argument('--str2idx', type=bool, default=False, help='string转为int')
	args = parser.parse_args()

	if args.action == 'fit':
		p = Preprocesser()
		p.fit(args.infilename)
		p.dump_sta_dict(args.stafilename)
	elif args.action == 'transform':
		p = Preprocesser(args.stafilename)
		p.transform(args.infilename, args.outfilename, str2idx=args.str2idx)
