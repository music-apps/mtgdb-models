import os
import glob
import hashlib

from collections import OrderedDict

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

class BaseLoader():

    # context structure for the tfrecord messages 
    tf_context = {
        'hash': tf.FixedLenFeature([], dtype=tf.string),
        'chunk_id': tf.FixedLenFeature([], dtype=tf.string),
        }

    # features structure for the tfrecord messages 
    tf_features = {
        'arrays': tf.VarLenFeature(dtype=tf.float32),
        'shapes': tf.VarLenFeature(dtype=tf.int64),
        'labels': tf.VarLenFeature(dtype=tf.string),
        'ids': tf.VarLenFeature(dtype=tf.string),
        }

    def __init__(self, filelist, groundtruth, output_folder, label='train', chunk_size=300., force=False, session=None):
        self.filelist = filelist
        self.groundtruth = groundtruth
        self.output_folder = output_folder
        self.label = label
        self.chunk_size = chunk_size
        self.force = force

        self.tf_records_filenames = []
        self.data_info = OrderedDict()

        self.check_input_files()
        self.parse_input_files()

        if not session:
            self.session = tf.InteractiveSession()
            self.owns_session = True
        else:
            self.session = session
            self.owns_session = False


    def __del__(self):
        if self.owns_session:
            self.session.close()


    @staticmethod
    def check_file(filename):
        assert os.path.exists(filename), 'Loader: file does not exist'

        with open(filename, 'r') as f:
            file_lines = f.readlines()

        keys = []
        for idx, line in enumerate(file_lines):
            entry = line.split('\t')

            assert len(entry) == 2, 'Line {} from filelist is not a valid entry.'.format(
                idx + 1)

            keys.append(entry[0])

        duplicates = set([x for x in keys if keys.count(x) > 1])

        assert not duplicates, 'Filelist contains {} repeated ids: {}'.format(
            len(duplicates), duplicates)
    
    
    @staticmethod
    def get_hash(s):
        return hashlib.sha224(''.join(s).encode('utf8')).hexdigest().encode('utf8')


    def path_hash(self):
        print('Hash: ' + ''.join([i['path'] for i in self.data_info.values()]))
        return self.get_hash(''.join([i['path'] for i in self.data_info.values()]))


    @staticmethod
    def parse_file(filename):
        with open(filename, 'r') as f:
            file_lines = f.readlines()

            file_dict = {k: v.rstrip() for (k, v) in [tuple(
                l.split('\t')) for l in file_lines]}
        return file_dict


    @staticmethod
    def dump_tfrecords(serialized, filename, force=False):
        if os.path.exists(filename) and not force:
            raise Exception('Loader: "{}" already exists'.format(filename))
        
        with tf.python_io.TFRecordWriter(filename) as writer:
                writer.write(serialized.SerializeToString())


    @staticmethod
    def _dtype_feature(data):
        if isinstance(data, np.ndarray):
            dtype_ = data.dtype
            if dtype_ == np.float64 or dtype_ == np.float32:
                return tf.train.Feature(float_list=tf.train.FloatList(value=data.flatten()))
            elif dtype_ == np.int64:
                return tf.train.Feature(int64_list=tf.train.Int64List(value=data.flatten()))
        elif isinstance(data, str):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.encode('utf-8')]))
        elif isinstance(data, bytes):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
        else:
            raise ValueError("The input is nopt a recognized type. \
                               Instaed got {}".format(type(data)))


    def load_tfrecords(self, filename):
        # Read and print data:
        #sess = tf.InteractiveSession()

        # Read TFRecord file
        reader = tf.TFRecordReader()
        filename_queue = tf.train.string_input_producer([filename])

        _, serialized_example = reader.read(filename_queue)

        # Extract features from serialized data
        context_data, features_data = tf.parse_single_sequence_example(serialized=serialized_example,
                                                                       context_features=self.tf_context,
                                                                       sequence_features=self.tf_features)

        # Many tf.train functions use tf.train.QueueRunner,
        # so we need to start it before we read
        tf.train.start_queue_runners(self.session)

        # # Print features
        # print('Context:')
        # for name, tensor in context_data.items():
        #     print('{}: {}'.format(name, tensor.eval()))

        # print('\nData')
        # print('ids: {}'.format(sequence_data['ids'].eval()))
        return context_data, features_data


    def check_input_files(self):
        # Check that a valid path i provided
        assert os.path.isdir(
            self.output_folder), 'Loader: invalid output folder'

        self.check_file(self.filelist)
        self.check_file(self.groundtruth)


    def parse_input_files(self):
        filelist = self.parse_file(self.filelist)
        groundtruth = self.parse_file(self.groundtruth)

        filelist_keys = filelist.keys()
        groundtruth_keys = groundtruth.keys()

        common_keys = set(filelist_keys).intersection(set(groundtruth_keys))
        common_n = len(common_keys)
        print('Loader: Groundtruth contains labels for {}/{} ids'.format(common_n,
                                                                         len(filelist_keys)))

        # ausume filelist contains relative paths to the .npys from filelist location
        base_path = os.path.dirname(self.filelist)
        self.data_info = OrderedDict({i: {'path': os.path.join(base_path, filelist[i]), 
                              'label': groundtruth[i]} for i in common_keys})


    def get_tfrecords(self):
        os.chdir(self.output_folder)
        glob_pattern = '*{}.tfrecords.*'.format(self.label)

        tf_records_filenames = [os.path.join(
            self.output_folder, f) for f in glob.glob(glob_pattern)]

        if(len(tf_records_filenames) == 0):
            print('no .tfrecords chunks found. Computing')
            
            self.npys_to_tfrecords()

        tf_records_filenames = [os.path.join(
            self.output_folder, f) for f in glob.glob(glob_pattern)]
        
        try:
            self.check_tfrecods(tf_records_filenames)
            
            # set the list of valid tf_records files
            self.tf_records_filenames = tf_records_filenames
            print('tfrecords files were sucessfully obtained')
    
        except Exception as e:
            print(e)
            print('All tfrecords files will be removed and computed again')

            for i in tf_records_filenames:
                os.remove(i)
            
            self.get_tfrecords()



    def check_tfrecods(self, tf_records_files):
        path_hash = self.path_hash()


        print('found {} tfrecords chuncks'.format(len(tf_records_files)))

        for f in tf_records_files:
            c, _ = self.load_tfrecords(f)
            chunk_hash = c['hash'].eval()
            chunk_id = int(c['chunk_id'].eval())

            assert(chunk_hash == path_hash), 'chunk number {} has an invalid hash: {}. Filelist hash: {}'.format(chunk_id, chunk_hash, path_hash)

        print('Correct hash for all found chunks')
        return tf_records_files


    def load_npy_instance(self, id, data, weight, ids):
        """
        This method defines the structure of the data inside
        the tfrecords files. It can be overrinded in a sibling
        class in order to modify the data shape without affecting
        the .npy to .tfrecord logic.
        """
        path = self.data_info[id]['path']
        label = self.data_info[id]['label']

        #initialization
        if not data:
            data['arrays'], data['shapes'], data['labels'], data['ids'] = [], [], [], []
        
        # Try to load the matrix
        try:
            array = np.load(path)
            shape = np.array(array.shape)
        except Exception as e:
            print('Error importing "{}" (id: {}). {}'.format(
                path, id, str(e)))
            return

        # update date
        data['arrays'].append(self._dtype_feature(array))
        data['shapes'].append(self._dtype_feature(shape))
        data['labels'].append(self._dtype_feature(label))
        data['ids'].append(self._dtype_feature(id))
        
        # update payload size
        weight += array.nbytes / 1e6 # store in megabytes
        
        # update list of successfully processed ids
        ids.append(id)


    def generate_context(self, c_hash, chunk_id):
        # TODO: Add info about features as the extractor, essentia version...
        context = {
            'hash': self._dtype_feature(c_hash),
            'chunk_id': self._dtype_feature(str(chunk_id))
            }

        return context


    def npys_to_tfrecords(self):
        remaining_keys = list(self.data_info.keys())
        n_total = len(remaining_keys)
        chunk_id = 0
        
        # make a hash with the processed paths
        path_hash = self.path_hash()
        
        while remaining_keys:
            weight = 0.0
            sucessful_ids = []
            data = dict()

            while((weight < self.chunk_size) and len(remaining_keys) > 0):
                self.load_npy_instance(remaining_keys[0], data,
                                       weight, sucessful_ids)
                remaining_keys.pop(0)

            # Context Features
            context = self.generate_context(path_hash, chunk_id)
            context = tf.train.Features(feature=context)
            
            # Sequential Features
            features = {k: tf.train.FeatureList(feature=v) for k, v in data.items()}
            features = tf.train.FeatureLists(feature_list=features)
            
            serialized = tf.train.SequenceExample(context=context,
                                                  feature_lists=features)
                
            # Write TFrecord
            writter_path = os.path.join(
                self.output_folder, self.label + '.tfrecords.{}'.format(chunk_id))

            self.dump_tfrecords(serialized, writter_path, force=self.force)
            print('Written {}/{}. Chunk {}.'.format(n_total -
                                                    len(remaining_keys), n_total, chunk_id))

            chunk_id += 1

