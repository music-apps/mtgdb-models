import yaml

from sklearn.preprocessing import LabelBinarizer

from . import *


class TsvToTFRecordLoader(BaseLoader):

    def __init__(self, filelist, groundtruth, output_folder,
                 label, config_file, label_binarizer=None,
                 force=False, session=None):

            # context structure for the tfrecord messages 
        self.tf_context = {
            'shape': tf.FixedLenFeature(2, dtype=tf.int64),
            'label_ohe': tf.FixedLenFeature([],  dtype=tf.int64),
            'label_tag': tf.FixedLenFeature([], dtype=tf.string),
            'id': tf.FixedLenFeature([], dtype=tf.string),
            }

        # features structure for the tfrecord messages 
        self.tf_features = {
            'data': tf.VarLenFeature(dtype=tf.float32),
            }

        self.filelist = filelist
        self.groundtruth = groundtruth
        self.output_folder = output_folder
    
        self.label = label
        self.force = force

        self.tfrecords_filenames = []
        self.data_info = dict()

        self.label_binarizer = label_binarizer

        if not session:
            self.session = tf.InteractiveSession()
            self.owns_session = True
        else:
            self.session = session
            self.owns_session = False

        try:
            self.config = yaml.load(open(config_file))
        except:
            raise('loader: Not able to load configuration file')

        self.chunk_size = self.config['loader']['chunk_size']

        self.chunk_size = self.config['loader']['chunk_size']

        # load these parameters and check that they fit the model.
        # otherwise print a warning and update project file.
        self.num_bands = self.config['architecture']['num_bands']
        self.num_classes = self.config['architecture']['num_classes']

        self.parse_input()
        self.get_tfrecords()

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
        paths = [i['path'] for i in self.data_info.values()]
        paths.sort()

        # print('Hash: ' + ''.join(paths))
        return self.get_hash(''.join(paths))

    @staticmethod
    def parse_file(filename):
        with open(filename, 'r') as f:
            file_lines = f.readlines()

            file_dict = {k: v.rstrip() for (k, v) in [tuple(
                l.split('\t')) for l in file_lines]}
        return file_dict

    @staticmethod
    def dump_tfrecords(serialized, writer, filename):
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

        return context_data, features_data


    def parse_input_files(self):
        # Check that a valid path i provided
        assert os.path.isdir(
            self.output_folder), 'Loader: invalid output folder'

        self.check_file(self.filelist)
        self.check_file(self.groundtruth)

        self.filelist_dict = self.parse_file(self.filelist)
        self.groundtruth_dict = self.parse_file(self.groundtruth)


    def parse_input(self):
        print('({}) loader: Parsing input files'.format(self.label))
        self.parse_input_files()

        filelist = self.filelist_dict
        groundtruth = self.groundtruth_dict

        filelist_keys = filelist.keys()
        groundtruth_keys = groundtruth.keys()

        common_keys = set(filelist_keys).intersection(set(groundtruth_keys))
        num_common = len(common_keys)
        print('({}) loader: Groundtruth contains labels for {}/{} ids in the filelist'.format(self.label, num_common,
                                                                                              len(filelist_keys)))

        self.num_tracks = num_common

        labels = [l for k, l in groundtruth.items() if k in common_keys]

        if not self.label_binarizer:
            print('({}) loader: warining. label_binarizer not available. Computing it from the labels on this set').format(
                self.label)
            self.label_binarizer = LabelBinarizer()
            self.label_binarizer.fit(labels)

        # update the number of classes
        n_classes = len(self.label_binarizer.classes_)
        self.tf_context['label_ohe'] = tf.FixedLenFeature(
            [n_classes],  dtype=tf.int64)

        assert n_classes == self.num_classes, '({}) loader: number of classes ({}) does not match the project file ({}).'.format(
            self.label, n_classes, self.num_classes)


        # ausume filelist contains relative paths to the .npys from filelist location
        base_path = os.path.dirname(self.filelist)
        self.data_info = {i: {'path': os.path.join(base_path, filelist[i]), 
                              'label_tag': groundtruth[i],
                              'label_ohe': self.label_binarizer.transform([groundtruth[i]])[0]} for i in common_keys}

    def get_tfrecords(self):
        os.chdir(self.output_folder)
        glob_pattern = '*{}.tfrecords.*'.format(self.label)

        tfrecords_filenames = [os.path.join(
            self.output_folder, f) for f in glob.glob(glob_pattern)]

        if(len(tfrecords_filenames) == 0):
            print('no .tfrecords chunks found. Computing')

            self.npys_to_tfrecords()

        tfrecords_filenames = [os.path.join(
            self.output_folder, f) for f in glob.glob(glob_pattern)]

        try:
            self.check_tfrecods(tfrecords_filenames)
            
            # set the list of valid tf_records files
            self.tfrecords_filenames = tfrecords_filenames
            print('tfrecords files were sucessfully obtained')
    
        except Exception as e:
            print(e)
            print('All tfrecords files will be removed and computed again')

            for i in tfrecords_filenames:
                os.remove(i)
            
            self.get_tfrecords()

    def check_tfrecods(self, tf_records_files):
        path_hash = self.path_hash()

        print('({}) loader: found {} tfrecords chuncks'.format(self.label, len(tf_records_files)))

        #for f in tf_records_files:
         #   c, _ = self.load_tfrecords(f)
            # chunk_hash = c['hash'].eval()
            # chunk_id = int(c['chunk_id'].eval())

            # assert(chunk_hash == path_hash), 'chunk number {} has an invalid hash: {}. Filelist hash: {}'.format(
            #     chunk_id, chunk_hash, path_hash)

        print('Correct hash for all found chunks')
        return tf_records_files


    def load_npy_instance(self, id):
        """This method defines the structure of the data inside
        the tfrecords files. It can be overrinded in a sibling
        class in order to modify the data shape without affecting
        the .npy to .tfrecord logic.
        """
        return np.load(self.data_info[id]['path'])



    def generate_context(self, shape, label_tag, label_ohe, id):
        # TODO: Add info about features as the extractor, essentia version...
        context = {
            'shape': self._dtype_feature(shape),
            'label_tag': self._dtype_feature(str(label_tag)),
            'label_ohe': self._dtype_feature(label_ohe),
            'id': self._dtype_feature(str(id))
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

            # Write TFrecord
            writter_path = os.path.join(
                self.output_folder, self.label + '.tfrecords.{}'.format(chunk_id))

            writer = tf.python_io.TFRecordWriter(writter_path)

            while((weight < self.chunk_size) and len(remaining_keys) > 0):
                id = remaining_keys[0]
                # print('writting id: {}'.format(id))

                label_tag = self.data_info[id]['label_tag']
                label_ohe = self.data_info[id]['label_ohe']

                # Try to load the matrix
                try:
                    data = self.load_npy_instance(remaining_keys[0])

                except Exception as e:
                    print('Error importing "{}". {}'.format(
                        remaining_keys[0], str(e)))
                    continue

                # update payload size
                weight += data.nbytes / 1e6 # store in megabytes

                # Context Features
                context = self.generate_context(np.array(data.shape), label_tag, label_ohe , id)
                context = tf.train.Features(feature=context)

                # Sequential Features
                features = {'data': tf.train.FeatureList(feature=[self._dtype_feature(data)])}
                features = tf.train.FeatureLists(feature_list=features)

                serialized = tf.train.SequenceExample(context=context,
                                                    feature_lists=features)

                writer.write(serialized.SerializeToString())

                # update list of successfully processed ids
                sucessful_ids.append(id)
                
                remaining_keys.pop(0)

            print('({}) loader: Written {}/{}. Chunk {}.'.format(self.label, n_total -
                                                    len(remaining_keys), n_total, chunk_id))

            writer.close()
            chunk_id += 1


# Class to evaluate files as provided by Evaluator.
# Fiiles are not checed as they are assumed to have been checked before
class YamlToTFRecordLoader(TsvToTFRecordLoader):

    def parse_input_files(self):
        
        import yaml
        
        # Check that a valid path i provided
        assert os.path.isdir(
            self.output_folder), '({}) loader: invalid output folder'.format(self.label)

        self.filelist_dict = yaml.load(open(self.filelist))
        self.groundtruth_dict = yaml.load(open(self.groundtruth))
