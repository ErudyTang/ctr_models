#coding=utf-8
import tensorflow as tf

class DNN(object):
	@staticmethod
	def _dnn(features, labels, mode, params):
		feature_columns = params['feature_columns']
		with tf.variable_scope('deep') as scope:
			fc = tf.feature_column.input_layer(features, feature_columns)
			for units in params['hidden_units']:
				fc = tf.layers.dense(fc, units=units,
					kernel_initializer=tf.glorot_normal_initializer,
					activation=tf.nn.relu)

		logits = tf.layers.dense(fc, params['n_classes'],
				kernel_initializer=tf.glorot_normal_initializer, activation=None)

		probabilities = tf.nn.softmax(logits)
		if mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {
				'probabilities': probabilities,
			}
			return tf.estimator.EstimatorSpec(mode, predictions=predictions)

		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		auc = tf.metrics.auc(labels, probabilities[:, 1])
		average_loss = tf.metrics.mean(loss)
		metrics = {'auc': auc, 'average_loss': average_loss}
		tf.summary.scalar('train_auc', auc[1])

		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = params['optimizer']
			train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

	@staticmethod
	def get_custom_estimator(model_dir, feature_columns, hidden_units):
		estimator_config = tf.estimator.RunConfig(
			model_dir=model_dir,
			keep_checkpoint_max=5,
			log_step_count_steps=1000,
		)
		opt = tf.train.AdagradOptimizer(0.1)

		params = {
				'feature_columns': feature_columns,
				'hidden_units': hidden_units,
				'n_classes': 2,
				'optimizer': opt,
			}

		return tf.estimator.Estimator(
			model_fn = DNN._dnn,
			params = params,
			config=estimator_config,
		)
	
	@staticmethod
	def get_estimator(model_dir, feature_columns, hidden_units):
		estimator_config = tf.estimator.RunConfig(
			model_dir=model_dir,
			keep_checkpoint_max=5,
			log_step_count_steps=1000,
		)
		opt = tf.train.AdagradOptimizer(0.01)

		return tf.estimator.DNNClassifier(
			hidden_units=hidden_units,
			feature_columns=feature_columns,
			optimizer=opt,
			activation_fn=tf.nn.relu,
			n_classes=2,
			config=estimator_config,
			#dropout = 0.5,
			#batch_norm=True,
		)

class LR(object):
	@staticmethod
	def get_estimator(model_dir, feature_columns):
		estimator_config = tf.estimator.RunConfig(
			model_dir=model_dir,
			keep_checkpoint_max=5,
			log_step_count_steps=1000,
		)
		opt = tf.train.FtrlOptimizer(0.01)

		return tf.estimator.LinearClassifier(
			feature_columns=feature_columns,
			optimizer=opt,
			n_classes=2,
			config=estimator_config,
		)

class WD(object):
	@staticmethod
	def get_estimator(model_dir, linear_feature_columns, dnn_feature_columns, hidden_units):
		estimator_config = tf.estimator.RunConfig(
			model_dir=model_dir,
			keep_checkpoint_max=5,
			log_step_count_steps=1000,
		)
		linear_opt = tf.train.FtrlOptimizer(0.01)
		dnn_opt = tf.train.AdagradOptimizer(0.01)

		return tf.estimator.DNNLinearCombinedClassifier(
			linear_feature_columns=linear_feature_columns,
			linear_optimizer = linear_opt,
			dnn_feature_columns=dnn_feature_columns,
			dnn_optimizer = dnn_opt,
			dnn_hidden_units = hidden_units,
			dnn_activation_fn=tf.nn.relu,
			n_classes=2,
			config=estimator_config,
		)

class FM(object):
	@staticmethod
	def _fm(features, labels, mode, params):
		linear_columns = params['linear_feature_columns']
		embedding_columns = params['embedding_feature_columns']

		with tf.variable_scope("fm"):
			linear_input = tf.feature_column.input_layer(features, linear_columns)
			linear_output = tf.reduce_sum(linear_input, axis=1, keepdims=True)

			embedding_inputs = []
			for column in embedding_columns:
				embedding_input = tf.feature_column.input_layer(features, column)
				embedding_inputs.append(embedding_input)
			sum_square = tf.add_n(embedding_inputs)  # 按位相加
			sum_square = tf.multiply(sum_square, sum_square)  # 按位相乘
			square_sums = []
			for embedding_input in embedding_inputs:
				square_sum = tf.multiply(embedding_input, embedding_input)
				square_sums.append(square_sum)
			square_sum = tf.add_n(square_sums)
			embedding_output = 0.5 * tf.subtract(sum_square, square_sum)

		fc = tf.concat([linear_output, embedding_output], axis=1)
		logits = tf.layers.dense(fc, params['n_classes'],
				kernel_initializer=tf.glorot_normal_initializer, activation=None)

		probabilities = tf.nn.softmax(logits)
		if mode == tf.estimator.ModeKeys.PREDICT:
			predictions = {
				'probabilities': probabilities,
			}
			return tf.estimator.EstimatorSpec(mode, predictions=predictions)

		loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
		auc = tf.metrics.auc(labels, probabilities[:, 1])
		average_loss = tf.metrics.mean(loss)
		metrics = {'auc': auc, 'average_loss': average_loss}
		tf.summary.scalar('train_auc', auc[1])

		if mode == tf.estimator.ModeKeys.EVAL:
			return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = params['optimizer']
			train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
	
	@staticmethod
	def get_estimator(model_dir, linear_feature_columns, embedding_feature_columns):
		estimator_config = tf.estimator.RunConfig(
			model_dir=model_dir,
			keep_checkpoint_max=5,
			log_step_count_steps=1000,
		)
		opt = tf.train.AdagradOptimizer(0.01)

		params = {
				'linear_feature_columns': linear_feature_columns,
				'embedding_feature_columns': embedding_feature_columns,
				'n_classes': 2,
				'optimizer': opt,
			}

		return tf.estimator.Estimator(
			model_fn = FM._fm,
			params = params,
			config=estimator_config,
		)
