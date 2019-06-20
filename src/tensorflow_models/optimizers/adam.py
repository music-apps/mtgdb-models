from . import *

class Adam(BaseOptimizer):

    def defiene_optimizer(self, model):
     epsilon = self.config['optimizer']['adam_epsilon']

     lr = tf.placeholder(tf.float32, name='learning_rate')
     y = tf.placeholder(tf.float32, [None, self.num_classes], name='onehot_labels')

     train_step = tf.Variable(
            0, name='train_step', trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
            tf.GraphKeys.GLOBAL_STEP])

     # Cross-entropy label loss.
     xent = tf.nn.sigmoid_cross_entropy_with_logits(
          logits=model.output, labels=y, name='xent')
     
     loss = tf.reduce_mean(xent, name='loss_op')
     tf.summary.scalar('loss', loss)

     # We use the same optimizer and hyperparameters as used to train VGGish.
     optimizer = tf.train.AdamOptimizer(
          learning_rate=lr,
          epsilon=epsilon)

     train_op = optimizer.minimize(loss, global_step=train_step, name='train_op')

     self.lr = lr
     self.y = y
     self.loss = loss
     self.train_step = train_step
     self.train_op = train_op
