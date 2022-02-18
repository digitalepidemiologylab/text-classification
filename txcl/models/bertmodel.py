"""
BERT
====
"""

from .base_model import BaseModel
from ..utils import viz
import csv
import logging
import os
import random
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional
from torch.optim import AdamW
from transformers import (
        AutoModelForSequenceClassification,
        AutoConfig,
        AutoTokenizer,
        # AdamW,
        get_linear_schedule_with_warmup)

logger = logging.getLogger(__name__)

class BERTModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.label_mapping = None
        self.train_examples = None
        self.num_train_optimization_steps = None

    def train(self, config):
        # Setup
        self._setup_bert(config)

        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if self.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            self.optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=self.learning_rate,
                                  bias_correction=False,
                                  eps = self.adam_epsilon,
                                  weight_decay=self.weight_decay_rate,
                                  max_grad_norm=1.0)
            if self.loss_scale == 0:
                self.optimizer = FP16_Optimizer(self.optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(self.optimizer, static_loss_scale=self.loss_scale)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon, weight_decay=self.weight_decay_rate)
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_train_optimization_steps)


        # Run training
        global_step = 0
        tr_loss = 0
        train_features = self.convert_examples_to_features(self.train_examples)
        logger.debug("***** Running training *****")
        logger.debug("  Num examples = %d", len(self.train_examples))
        logger.debug("  Batch size = %d", self.train_batch_size)
        logger.debug("  Num steps = %d", self.num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if self.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.train_batch_size)
        loss_vs_time = []
        for epoch in range(int(self.num_epochs)):
            self.model.train()
            nb_tr_examples, nb_tr_steps = 0, 0
            epoch_loss = 0
            pbar = tqdm(train_dataloader)
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                outputs = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                loss = outputs[0]
                logits = outputs[1]
                if self.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps
                if self.fp16:
                    self.optimizer.backward(loss)
                else:
                    loss.backward()
                loss = loss.item()
                tr_loss += loss
                epoch_loss += loss
                if step > 0:
                    pbar.set_description("Loss: {:8.4f} | Average loss/it: {:8.4f}".format(loss, epoch_loss/step))
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
            # evaluate model
            if self.eval_after_epoch:
                self.model.eval()
                nb_train_steps, nb_train_examples = 0, 0
                train_accuracy, train_loss = 0, 0
                for input_ids, input_mask, segment_ids, label_ids in tqdm(train_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(self.device)
                    input_mask = input_mask.to(self.device)
                    segment_ids = segment_ids.to(self.device)
                    label_ids = label_ids.to(self.device)
                    with torch.no_grad():
                        loss, logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
                    train_accuracy += self.accuracy(logits.to('cpu').numpy(), label_ids.to('cpu').numpy())
                    train_loss += loss.mean().item()
                    nb_train_examples += input_ids.size(0)
                    nb_train_steps += 1
                train_loss = train_loss / nb_train_steps
                train_accuracy = 100 * train_accuracy / nb_train_examples
                print("{bar}\nEpoch {}:\nTraining loss: {:8.4f} | Training accuracy: {:.2f}%\n{bar}".format(epoch+1, train_loss, train_accuracy, bar=80*'='))

        __import__('pdb').set_trace()
        # Save model
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        logger.info(f'Saving model to {self.output_path}...')
        model_to_save.save_pretrained(self.output_path)

    def test(self, config):
        # Setup
        self._setup_bert(config, setup_mode='test')
        # Run test
        eval_examples = self.processor.get_dev_examples(self.test_path)
        eval_features = self.convert_examples_to_features(eval_examples)
        logger.debug("***** Running evaluation *****")
        logger.debug("  Num examples = %d", len(eval_examples))
        logger.debug("  Batch size = %d", self.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        result = {'prediction': [], 'label': [], 'text': []}
        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            tmp_eval_loss, logits = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            result['prediction'].extend(np.argmax(logits, axis=1).tolist())
            result['label'].extend(label_ids.tolist())
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        result_out = self.performance_metrics(result['label'], result['prediction'], label_mapping=self.label_mapping)
        if self.write_test_output:
            test_output = self.get_full_test_output(result['prediction'], result['label'], label_mapping=self.label_mapping,
                    test_data_path=self.test_path)
            result_out = {**result_out, **test_output}
        return result_out

    def predict(self, config, data):
        """Predict data (list of strings)"""
        # Setup
        self._setup_bert(config, setup_mode='predict', data=data)
        # Run predict
        predict_examples = self.processor.get_test_examples(data)
        predict_features = self.convert_examples_to_features(predict_examples)
        all_input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in predict_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in predict_features], dtype=torch.long)
        predict_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        predict_sampler = SequentialSampler(predict_data)
        predict_dataloader = DataLoader(predict_data, sampler=predict_sampler, batch_size=self.eval_batch_size)
        self.model.eval()
        result = []
        for input_ids, input_mask, segment_ids, label_ids in predict_dataloader:
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            output = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)
            logits = output[0]
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            probabilities = probabilities.detach().cpu().numpy()
            res = self.format_predictions(probabilities, label_mapping=self.label_mapping)
            result.extend(res)
        return result

    def _setup_bert(self, config, setup_mode='train', data=None):
        # Paths
        self.model_path = os.path.join(config.path.other, 'bert')
        self.output_path = config.path.output
        self.train_path = config.data.train
        self.test_path = config.data.test
        if config.model.params.get('pretrain_name', None):
            self.pretrain_path = config.model.params.get('pretrain_model_path', os.path.join(config.path.other, 'pretrain', 'bert', config.model.params.get('pretrain_name', None)))
        else:
            self.pretrain_path = None

        # Hyperparams
        self.max_seq_length = config.model.params.get('max_seq_length', 96)
        self.train_batch_size = config.model.params.get('train_batch_size', 32)
        self.eval_batch_size = config.model.params.get('eval_batch_size', 32)
        # Initial learning rate for Adam optimizer
        self.learning_rate = config.model.params.get('learning_rate', 5e-5)
        self.num_epochs = int(config.model.params.get('num_epochs', 3))
        # Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_accumulation_steps = config.model.params.get('gradient_accumulation_steps', 1)
        # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.
        self.warmup_steps = int(self.num_epochs * len(pd.read_csv(self.train_path)) * config.model.params.get('warmup_proportion', 0.1) / (self.train_batch_size * self.gradient_accumulation_steps))
        self.no_cuda = config.model.params.get('no_cuda', False)
        #local_rank for distributed training on gpus
        self.local_rank = config.model.params.get('local_rank', -1) # disable distributed training by setting local_rank = -1
        self.seed = config.model.params.get('seed', np.random.randint(1e4))
        # Use 16 bit float precision (instead of 32bit)
        self.fp16 = config.model.params.get('fp16', False)
        # Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.
        # 0 (default value): dynamic loss scaling. Positive power of 2: static loss scaling value.
        self.loss_scale = config.model.params.get('loss_scale', 0.0)

        self.adam_epsilon = config.model.params.get('adam_epsilon', 1e-8)
        self.eval_after_epoch = config.model.params.get('eval_after_epoch', True)
        self.weight_decay_rate = config.model.params.get('weight_decay_rate', 1e-4)
        # Meta params
        self.output_attentions = config.model.params.get('output_attentions', False)
        self.write_test_output = config.write_test_output
        # self.use_pretrain_model = config.get('use_pretrain_model', False)
        # self.run_index = config.get('run_index', 0)
        # self.learning_curve_fraction = config.get('learning_curve_fraction', 0)

        # GPU config
        if not torch.cuda.is_available() or self.no_cuda:
            self.device = torch.device("cpu")
            self.n_gpu = 0
        else: # cuda is available and we want to use it
            if self.local_rank == -1:
                self.device = torch.device("cuda")
                self.n_gpu = torch.cuda.device_count()
            else: # Distributed Data Parallel (DDP)
                torch.cuda.set_device(self.local_rank)
                self.n_gpu = 1
                torch.distributed.init_process_group(backend='nccl')

        if setup_mode == 'train': 
            logger.info("Initialize BERT: device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(self.device, self.n_gpu, bool(self.local_rank != -1), self.fp16))
        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(self.gradient_accumulation_steps))
        self.train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        # seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)

        # label mapping
        if setup_mode == 'train':
            self.label_mapping = self._set_label_mapping(self.train_path, self.test_path, self.output_path)
        elif setup_mode in ['test', 'predict']:
            self.label_mapping = self.load_label_mapping(self.output_path)

        # Build model
        self.processor = SentimentClassificationProcessor(self.train_path, self.label_mapping)
        num_labels = len(self.label_mapping)
        self.model_type = config.model.params.get('model_type', 'bert-base-uncased')
        self.do_lower_case = 'uncased' in self.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type, do_lower_case=self.do_lower_case)
        if setup_mode == 'train':
            self.train_examples = self.processor.get_train_examples(self.train_path)
            self.num_train_optimization_steps = int(len(self.train_examples) / self.train_batch_size / self.gradient_accumulation_steps) * self.num_epochs
            if self.local_rank != -1:
                self.num_train_optimization_steps = self.num_train_optimization_steps // torch.distributed.get_world_size()

        # Prepare model
        if setup_mode == 'train':
            if self.pretrain_path:
                logger.info('Loading pretrain model {} of type {}...'.format(self.pretrain_path, self.model_type))
                config = AutoConfig.from_pretrained(self.pretrain_path, num_labels=num_labels, label2id=self.label_mapping)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.pretrain_path,
                        cache_dir=self.model_path, # Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
                        config=config) 
            else:
                logger.info('Loading pretrained model {}...'.format(self.model_type))
                config = AutoConfig.from_pretrained(
                        self.model_type,
                        num_labels=num_labels,
                        label2id=self.label_mapping,
                        id2label={str(v): k for  k, v in self.label_mapping.items()})
                self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_type,
                        cache_dir=self.model_path,
                        config=config)
            if self.fp16:
                self.model.half()
        else:
            # Load a trained model and config that you have trained
            config = AutoConfig.from_pretrained(self.output_path, output_attentions=self.output_attentions)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.output_path,
                    config=config) 
        self.model.to(self.device)
        if self.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            self.model = DDP(self.model)
        elif self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def accuracy(self, out, labels):
        outputs = np.argmax(out, axis=1)
        return np.sum(outputs == labels)

    def convert_examples_to_features(self, examples):
        """Loads a data file into a list of ``InputBatch`` s."""
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = self.tokenizer.tokenize(str(example.text_a))
            tokens_b = None
            if example.text_b:
                tokens_b = self.tokenizer.tokenize(str(example.text_b))
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > self.max_seq_length - 2:
                    tokens_a = tokens_a[:(self.max_seq_length - 2)]
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)
            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            # Zero-pad up to the sequence length.
            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            assert len(input_ids) == self.max_seq_length
            assert len(input_mask) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            label_id = self.label_mapping[example.label]
            if ex_index < 5:
                logger.debug("*** Example ***")
                logger.debug("guid: %s" % (example.guid))
                logger.debug("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.debug("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.debug(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.debug("label: %s (id = %d)" % (example.label, label_id))
            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
        return features

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()
    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    @classmethod
    def _read_csv(cls, input_file):
        """Reads a pandas dataframe"""
        return pd.read_csv(input_file)

class SentimentClassificationProcessor(DataProcessor):
    """Processor for the sentiment classification data set."""
    def __init__(self, train_path, labels):
        self.labels = labels
        self.train_path = train_path

    def train_validation_split(self, validation_size=0.1):
        with open(self.train_path) as f:
            num_train_examples =  sum(1 for line in f) - 1
            ids = np.arange(num_train_examples)
            np.random.shuffle(ids)
            split_id = int(num_train_examples*validation_size)
            return ids[:split_id], ids[split_id:]

    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_csv(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_csv(data_path), "dev")

    def get_test_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if isinstance(lines, list):
            lines = pd.DataFrame({'text': lines})
        for (i, line) in lines.iterrows():
            guid = "%{}-{}".format(set_type, i)
            text = line['text']
            if set_type == "test":
                label = list(self.labels)[0]
            else:
                label = line['label']
            examples.append(InputExample(guid=guid, text_a=text, text_b=None, label=label))
        return examples
