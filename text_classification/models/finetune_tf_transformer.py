from .base_model import BaseModel
from text_classification.utils.transformers_helpers import mask_tokens, rotate_checkpoints, set_seed, download_vocab_files_for_tokenizer
from transformers import (
    AutoConfig,
    TFAutoModelWithLMHead,
    AutoTokenizer
)
from tqdm import tqdm
import logging
import os
import numpy as np
import torch
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
import glob
import collections

logger = logging.getLogger(__name__)

USE_XLA = False
USE_AMP = False
tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})


class FinetuneTfTransformer(BaseModel):
    def __init__(self):
        super().__init__()

    def init(self, config):
        # Paths
        self.name = config.name
        self.train_data = config.train_data
        self.test_data = config.get('test_data', None)
        self.other_path = config.other_path
        self.output_path = config.output_path
        self.tmp_path = config.tmp_path
        self.model_name = config.get('model', 'bert')
        self.model_type = config.get('model_type', 'bert-base-uncased')
        self.model_path = os.path.join(self.other_path, self.model_name)
        self.overwrite = config.get('overwrite', False)
        self.load_data_into_memory = config.get('load_data_into_memory', False)
        self.num_workers_batch_loading = config.get('num_workers_batch_loading', 50) # only used when load_data_into_memory is False
        self.evaluate_during_training = config.get('evaluate_during_training', True)

        # config
        self.mlm = self.model_name in ["bert", "roberta", "distilbert", "camembert"]  # is masked LM
        self.mlm_probability = config.get('mlm_probability', 0.15)
        self.save_steps = config.get('save_steps', 500)  # save every n steps
        self.num_checkpoints = config.get('num_checkpoints', 1)  # save n last checkpoints (set to 1 due to large model files)

        # hyperparams
        self.max_seq_length = config.get('max_seq_length', 128)
        self.train_batch_size = config.get('train_batch_size', 16)
        self.test_batch_size = config.get('test_batch_size', 16)
        self.learning_rate = config.get('learning_rate', 5e-5)
        self.epsilon = config.get('epsilon', 1e-8)
        self.num_epochs = config.get('num_epochs', 1)
        self.warmup_steps = config.get('warmup_steps', 100)
        self.max_train_steps = config.get('max_train_steps', None)
        self.no_cuda = config.get('no_cuda', False)
        self.on_memory = config.get('on_memory', True)
        self.do_lower_case = 'uncased' in self.model_type
        self.local_rank = config.get('local_rank', -1)
        self.seed = config.get('seed', np.random.randint(1e4))
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.fp16 = config.get('fp16', False)
        self.loss_scale = config.get('loss_scale', 0.0)

        # set seed
        set_seed(self.seed, no_cuda=self.no_cuda)

    def train(self):
        # load default tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_type, cache_dir=self.model_path)
        # load fast tokenizer
        vocab_files = download_vocab_files_for_tokenizer(tokenizer, self.model_type, self.output_path)
        fast_tokenizer = BertWordPieceTokenizer(vocab_files.get('vocab_file'), vocab_files.get('merges_file'), lowercase=self.do_lower_case)
        fast_tokenizer.enable_padding(max_length=self.max_seq_length)
        # Load traning data
        logger.debug(f'Generating TF record files...')

        # crate tf record datasets
        self.create_tf_record_datasets(fast_tokenizer, tokenizer)

        def parse_records(record):
            def decode_record(record, name_to_features):
                """Decodes a record to a TensorFlow example."""
                example = tf.io.parse_single_example(record, name_to_features)
                # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
                # So cast all int64 to int32.
                for name in list(example.keys()):
                    t = example[name]
                    if t.dtype == tf.int64:
                        t = tf.cast(t, tf.int32)
                    example[name] = t
                return {'input_ids': example['input_ids'], 'attention_mask': example['attention_mask']}, example['targets']
            int_feature = tf.io.FixedLenFeature([self.max_seq_length], tf.int64)
            name_to_features = {
                'input_ids': int_feature,
                'attention_mask': int_feature,
                'targets': int_feature
            }
            return decode_record(record, name_to_features)

        dataset = tf.data.TFRecordDataset(glob.glob(os.path.join(self.output_path, 'train_*.tfrecords')))
        dataset = dataset.map(parse_records)
        dataset = dataset.batch(self.train_batch_size)
        # load model
        config = AutoConfig.from_pretrained(self.model_type, cache_dir=self.model_path)
        model = TFAutoModelWithLMHead.from_pretrained(
            self.model_type,
            config=config,
            cache_dir=self.model_path)
        # optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, epsilon=self.epsilon)
        if USE_AMP:
            # loss scaling is currently required when using mixed precision
            optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, 'dynamic')
        # loss function
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # compile mode
        model.compile(optimizer=optimizer, loss=loss)
        # fit model
        history = model.fit(dataset, epochs=1, steps_per_epoch=int(1000/self.train_batch_size))
        # save model
        model.save_pretrained(self.output_path)


    def create_tf_record_datasets(self, fast_tokenizer, tokenizer):
        def generate_example():
            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f
            logger.info(f'Reading file {self.train_data}')
            num_lines = sum(1 for _ in open(self.train_data, 'r'))
            with tf.io.gfile.GFile(self.train_data, 'r') as reader:
                while True:
                    line = reader.readline()
                    if not line:
                        break
                    if len(line.strip()) == 0:
                        continue
                    # padding
                    encoding = tokenizer.encode_plus(line, max_length=self.max_seq_length, pad_to_max_length=True)
                    inputs = encoding['input_ids']
                    # use pytorch to mask tokens
                    inputs = torch.tensor([inputs], dtype=torch.long)
                    inputs, labels = mask_tokens(inputs, tokenizer, mlm_probability=self.mlm_probability) if self.mlm else (batch, batch)
                    features = collections.OrderedDict()
                    features['input_ids'] = create_int_feature(inputs[0])
                    features['attention_mask'] = create_int_feature(encoding['attention_mask'])
                    features['targets'] = create_int_feature(labels[0])
                    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                    serialized_example = tf_example.SerializeToString()
                    yield serialized_example

        serialized_examples = generate_example()
        output_path = os.path.join(self.output_path, 'train_000.tfrecords')
        logger.info(f'Writing tfrecords file {output_path}...')
        with tf.io.TFRecordWriter(output_path) as writer:
            for i, serialized_example in enumerate(serialized_examples):
                writer.write(serialized_example)
                # To be removed: For testing purposes only
                if i > 1000:
                    break
        logger.info(f'...done')


    def test(self):
        pass
