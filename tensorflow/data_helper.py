# coding=utf-8
import tensorflow as tf

class TextDataHelper(object):
    def __init__(self, _config, _parallel_num=40):
        self.config = _config
        self.parallel_num = _parallel_num
        self.csv_column_keys = [self.config.label.name]
        self.csv_defaults = [[self.config.label.default]]
        for fea in self.config.features:
            self.csv_column_keys.append(fea.name)
            self.csv_defaults.append([fea.default])

    def _parse_line(self, line):
        fields = tf.io.decode_csv(line, record_defaults=self.csv_defaults, use_quote_delim=False, field_delim='\t')
        features = dict(zip(self.csv_column_keys, fields))
        # 特殊处理下|分割的list，模型部署时注意这里的预处理不在模型内部
        for k in self.config.need_split_features:
            features[k] = tf.string_split(features[k], sep='|')
        label = features.pop(self.config.label.name)
        return features, label

    def train_input_fn(self, file_names, batch_size, shuffle=True):
        dataset = tf.data.TextLineDataset(file_names)
        if shuffle:
            dataset = dataset.shuffle(1000000)
        dataset = dataset.batch(batch_size).map(self._parse_line, num_parallel_calls=self.parallel_num)
        dataset = dataset.prefetch(1)
        return dataset
