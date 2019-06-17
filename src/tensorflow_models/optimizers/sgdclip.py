from . import *

class SGDClip(BaseOptimizer):

    def defiene_optimizer(self, model):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batchnorm
        
        with tf.control_dependencies(update_ops):
            lr = tf.placeholder(tf.float32, name='learning_rate')

            y = tf.placeholder(tf.float32, [None, self.num_classes], name='onehot_labels')

            loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=model.output)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

            gradients, variables = zip(*optimizer.compute_gradients(loss))

            gradients, _ = tf.clip_by_global_norm(gradients, 5.0)

            train_step = optimizer.apply_gradients(zip(gradients, variables))
            
            self.lr = lr
            self.y = y
            self.loss = loss
            self.train_step = train_step