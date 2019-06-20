from __future__ import absolute_import, division, print_function

from abc import ABC, abstractclassmethod
import random
import time
import yaml
import os 

import numpy as np
import tensorflow as tf
import pescador as ps
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class BaseTrainer(ABC):
    eps = np.finfo(float).eps

    def __init__(self, model, train_loader, optimizer, config_file,
                 val_loader=None, session=None):
        super().__init__()

        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer

        if not session:
            self.session = tf.InteractiveSession()
            self.owns_session = True
        else:
            self.session = session
            self.owns_session = False

        try:
            self.config = yaml.load(open(config_file))
        except:
            raise('BaseTrainer: Not able to load configuration file')

        self.num_epochs = self.config['trainer']['num_epochs']
        self.subset_size = self.config['trainer']['subset_size']
        self.batch_size = self.config['trainer']['batch_size']
        self.sampling_strategy = self.config['trainer']['sampling_strategy']
        self.batch_hop_size = self.config['trainer']['batch_hop_size']
        
        self.num_frames = self.config['architecture']['num_frames']
        self.num_bands = self.config['architecture']['num_bands']
        self.num_classes = self.config['architecture']['num_classes']

        self.inital_lr = self.config['optimizer']['learning_rate']

        if val_loader:
            assert type(train_loader) == type(val_loader)
            self.val_loader = val_loader

        self.initialize()


    @abstractclassmethod
    def initialize(self):
        pass

    @abstractclassmethod
    def train(self):
        pass

    def _parse_single_sequence_example(self, data_record):
        context, data =  tf.parse_single_sequence_example(data_record,
                                                context_features=self.train_loader.tf_context,
                                                sequence_features=self.train_loader.tf_features)
        return data['data'].values, context['label_ohe']

    @staticmethod
    def _data_gen(x, y, num_frames, batch_hop_size, strategy):
        # let's deliver some data!
        last_frame = int(x.shape[0]) - int(num_frames) + 1
        if strategy == 'single_shot':
            time_stamp = random.randint(0,last_frame - 1)
            yield {'X': x[time_stamp : time_stamp + num_frames, :], 'Y': y}

        elif strategy == 'overlapped_sampling':
            for time_stamp in range(0, last_frame, batch_hop_size):
                yield {'X': x[time_stamp : time_stamp + num_frames, :], 'Y': y}

        else:
            raise Exception('Sampling strategy not implemented')


class Trainer(BaseTrainer):

    def initialize(self):
        # Shape for the padded batch
        shapes = (tf.TensorShape([None,]), tf.TensorShape([None,]))

        train_dataset = tf.data.TFRecordDataset(self.train_loader.tfrecords_filenames)
        train_dataset = train_dataset.map(self._parse_single_sequence_example)  # Parse the record into tensors.
        
        train_dataset = train_dataset.padded_batch(self.subset_size, padded_shapes=shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.train_next_element = self.train_iterator.get_next()


        val_dataset = tf.data.TFRecordDataset(self.val_loader.tfrecords_filenames)
        val_dataset = val_dataset.map(self._parse_single_sequence_example)  # Parse the record into tensors.

        # val_subset_size = np.min([self.subset_size, self.val_loader.num_tracks])
        val_dataset = val_dataset.padded_batch(self.subset_size, padded_shapes=shapes)

        self.val_iterator = val_dataset.make_initializable_iterator()
        self.val_next_element = self.val_iterator.get_next()

    def _get_batch_generator(self, next_element):
        x_batch, y_batch = self.session.run(next_element)

        batch_size = len(x_batch)

        # Get tensors in shape
        shape =  np.array([-1, self.num_bands])
        x_batch = [np.reshape(x, shape) for x in x_batch]
        
        # Add singleton dimension for channel
        x_batch = np.expand_dims(x_batch, -1)
        
        # With 2 classes labels are stores in 1D binary array
        # so an extra singleton dimension is needed
        if self.num_classes == 2:
            y_batch = np.expand_dims(y_batch, -1)


        # Use pescador to generate the bathches
        # TODO: training batch generator shoulrain_batch_streamer =d be into the data loader
        train_streams = [ps.Streamer(self._data_gen, x, y, self.num_frames,
                                     self.batch_hop_size, self.sampling_strategy) for x, y in zip(x_batch, y_batch)]

        train_mux = ps.StochasticMux(train_streams,
                                     n_active=len(train_streams),
                                     rate=None,
                                     mode='exhaustive')

        return ps.Streamer(ps.buffer_stream,
                           train_mux,
                           buffer_size=batch_size,
                           partial=True)


    def train(self):
        print('trainer: training loop')
 
        num_subsets = int(np.ceil(self.train_loader.num_tracks / self.subset_size))
        print('trainer: training on {} tracks on {} subsets of {} tracks each.'.format(self.train_loader.num_tracks,
                                                                                       num_subsets, self.subset_size))
        
        num_subsets_val = int(np.ceil(self.val_loader.num_tracks / self.subset_size))
        print('trainer: validating on {} tracks on {} subsets of {} tracks each.'.format(self.val_loader.num_tracks,
                                                                                         num_subsets_val, self.subset_size))

        sess = self.session
        current_lr = self.inital_lr

        sess.run(tf.global_variables_initializer())

        for epoch_n in range(self.num_epochs):
            print('\n')
            print('*' * 24)
            print('epoch {}/{} starting...'.format(epoch_n + 1, self.num_epochs))
            print('*' * 24)

            start_time = time.time()
            array_train_loss = []
            array_train_acc = []

            idx_subset = 1

            sess.run(self.train_iterator.initializer)


            while True:
                try:
                    for batch_num, train_batch in enumerate(self._get_batch_generator(self.train_next_element)):
                        _, _, train_loss, preds = sess.run([self.optimizer.train_step, self.optimizer.train_op, self.optimizer.loss, self.model.normalized_output],
                                                feed_dict={self.model.input: train_batch['X'],
                                                            self.optimizer.y: train_batch['Y'],
                                                            self.model.is_training: True,
                                                            self.optimizer.lr: current_lr})

                        print('batch {} done. ({} chunks)'.format(batch_num + 1, train_batch['X'].shape[0]))

                        array_train_loss.append(train_loss)

                        preds_argmax = np.argmax(np.array(preds), axis=1)
                        y_argmax = np.argmax(train_batch['Y'], axis=1)
                        acc = accuracy_score(preds_argmax, y_argmax)
                        array_train_acc.append(acc)

                except tf.errors.OutOfRangeError:
                    break

                subset_loss = np.mean(array_train_loss)
                print('subset {}/{} done. (loss: {:.3f})'.format(idx_subset, num_subsets, subset_loss))
                idx_subset += 1

            train_loss = np.mean(array_train_loss)
            train_acc = np.mean(array_train_acc)


            print('validating...')
            sess.run(self.val_iterator.initializer)
            
            array_val_acc, array_val_loss = [], []

            while True:
                try:
                    for batch_num, val_batch in enumerate(self._get_batch_generator(self.val_next_element)):
                        preds = sess.run([self.model.normalized_output],
                                         feed_dict={self.model.input: val_batch['X'],
                                                    self.model.is_training: False})

                        val_loss = sess.run([self.optimizer.loss],
                                            feed_dict={self.model.input: val_batch['X'],
                                                       self.optimizer.y: val_batch['Y'],
                                                       self.model.is_training: False})

                        print('batch {} ({} chunks) done'.format(batch_num + 1, val_batch['X'].shape[0]))
                        print('val loss : {}'.format(val_loss))
                        array_val_loss.append(val_loss)
                        
                        preds_argmax = np.argmax(np.array(preds[0]), axis=1)
                        y_argmax = np.argmax(val_batch['Y'], axis=1)

                        acc = accuracy_score(preds_argmax, y_argmax)

                        array_val_acc.append(acc)

                except tf.errors.OutOfRangeError:
                    break
                    
            #Keep track of average loss of the epoch
            epoch_time = time.time()

            val_loss = np.mean(array_val_loss)
            val_acc = np.mean(array_val_acc)

            print('train_loss: {:.3f} train_acc: {:.3f} val_loss: {:.3f} val_acc: {:.3f} lr: {:.5f} time_s: {:.3f}'.format(
                train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time - start_time))

            results_dir = self.config['results_directory']
            fy = open(os.path.join(results_dir, 'train_log.tsv'), 'a')
            fy.write('%d\t%g\t%g\t%g\t%gs\t%g\n' % (epoch_n + 1, train_loss, val_loss, val_acc, epoch_time, current_lr))

            
                    # Early stopping: keep the best model in validation set
                    # if acc_best_model > accuracy:
                    #     print('Epoch %d, train cost %g, val cost %g, accuracy %g, '
                    #           'epoch-time %gs, lr %g, time-stamp %s' %
                    #           (i+1, train_loss, val_loss, accuracy, epoch_time, current_lr,
                    # self.(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()))))
                    
                    # else:
                    #     # save model weights to disk
                    #     save_path = saver.save(sess, model_folder)
                    #     print('Epoch %d, train cost %g, val cost %g, accuracy %g, '
                    #           'epoch-time %gs, lr %g, time-stamp %s - [BEST MODEL]'
                    #           ' saved in: %s' %
                    #           (i+1, train_loss, val_loss, accuracy, epoch_time,current_lr,
                    # self.(time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())), save_path))
                    #     cost_best_model = val_loss
                    # acc_best_model = accuracy
