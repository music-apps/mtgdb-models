from . import *


class VGGish(BaseModel):

    def define_model(self,):
        with tf.variable_scope('vggish'):
            input_layer = self.input
            num_filters = self.config['architecture']['num_filters']

            bn_input = tf.layers.batch_normalization(input_layer, training=self.is_training, axis=-1, renorm=True)
            kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True)

            conv1 = tf.layers.conv2d(inputs=bn_input,
                                    filters=num_filters,
                                    kernel_size=[3, 3],
                                    padding='same',
                                    activation=tf.nn.elu,
                                    name='1CNN',
                                    kernel_initializer=kernel_initializer)
            bn_conv1 = tf.layers.batch_normalization(conv1, training=self.is_training, axis=-1, renorm=True)
            pool1 = tf.layers.max_pooling2d(inputs=bn_conv1, pool_size=[2, 2], strides=[2, 2])
            print(pool1.get_shape)

            conv2 = tf.layers.conv2d(inputs=pool1,
                                    filters=num_filters,
                                    kernel_size=[3, 3],
                                    padding='same',
                                    activation=tf.nn.elu,
                                    name='2CNN',
                                    kernel_initializer=kernel_initializer)
            bn_conv2 = tf.layers.batch_normalization(conv2, training=self.is_training, axis=-1, renorm=True)
            pool2 = tf.layers.max_pooling2d(inputs=bn_conv2, pool_size=[2, 2], strides=[2, 2])
            print(pool2.get_shape)

            conv3 = tf.layers.conv2d(inputs=pool2,
                                    filters=num_filters,
                                    kernel_size=[3, 3],
                                    padding='same',
                                    activation=tf.nn.elu,
                                    name='3CNN',
                                    kernel_initializer=kernel_initializer)
            bn_conv3 = tf.layers.batch_normalization(conv3, training=self.is_training, axis=-1, renorm=True)
            pool3 = tf.layers.max_pooling2d(inputs=bn_conv3, pool_size=[2, 2], strides=[2, 2])
            print(pool3.get_shape)

            conv4 = tf.layers.conv2d(inputs=pool3,
                                    filters=num_filters,
                                    kernel_size=[3, 3],
                                    padding='same',
                                    activation=tf.nn.elu,
                                    name='4CNN',
                                    kernel_initializer=kernel_initializer)
            bn_conv4 = tf.layers.batch_normalization(conv4, training=self.is_training, axis=-1, renorm=True)
            pool4 = tf.layers.max_pooling2d(inputs=bn_conv4, pool_size=[2, 2], strides=[2, 2])
            print(pool4.get_shape)

            conv5 = tf.layers.conv2d(inputs=pool4, 
                                    filters=num_filters, 
                                    kernel_size=[3, 3], 
                                    padding='same', 
                                    activation=tf.nn.elu,
                                    name='5CNN', 
                                    kernel_initializer=kernel_initializer)
            bn_conv5 = tf.layers.batch_normalization(conv5, training=self.is_training, axis=-1, renorm=True)
            pool5 = tf.layers.max_pooling2d(inputs=bn_conv5, pool_size=[2, 2], strides=[2, 2])
            print(pool5.get_shape)

            flat = tf.layers.flatten(pool5)
            do = tf.layers.dropout(flat, rate=0.5, training=self.is_training)

            print(do.get_shape)
            output = tf.layers.dense(inputs=do,
                                activation=None,
                                units=self.num_classes,
                                kernel_initializer=kernel_initializer)

            normalized_output = tf.sigmoid(output, name='prediction')

            self.output = output
            self.normalized_output = normalized_output
