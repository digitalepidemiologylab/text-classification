from models.base_model import BaseModel
from tqdm import trange, tqdm
import torch
import torch.nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
from utils.misc import suppress_stdout
with suppress_stdout():
    from pytorch_pretrained_bert import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer
import os
import pandas as pd

class OpenAIGPT2(BaseModel):
    def __init__(self):
        super().__init__()
        self.transformer = None
        self.transformer_lm = None
        self.linear_model = None
        self.tokenizer = None

    def init_params(self, config, setup_mode='train'):
        # model params
        self.batch_size = config.get('batch_size', 8)
        self.test_batch_size = config.get('test_batch_size', 8)
        self.temperature = config.get('temperature', 0.9)
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.num_epochs = int(config.get('num_epochs', 3))
        # paths
        if len(config) == 0:
            self.model_save_path = '.'
            self.model_path = os.path.join('.', 'data', 'other', 'openai_gpt2')
        else:
            self.model_save_path = os.path.join(config.output_path, 'linear_model.pt')
            self.model_path = os.path.join(config.other_path, 'openai_gpt2')
        # GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # seed
        self.update_seed()
        # label mapping
        if setup_mode == 'train':
            self.label_mapping = self.set_label_mapping(config)
        elif setup_mode == 'generate_text':
            pass
        else:
            self.label_mapping = self.get_label_mapping(config)

    def train(self, config):
        # Init params and models
        self.init_params(config)
        self.load_models(transformer_only=True)

        # Load data
        dataset = MyDataset(config.train_data, transform=Encoder(self.tokenizer, self.label_mapping, device=self.device, max_size=self.transformer.config.n_positions))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Linear model (classifier)
        input_dim = self.transformer.config.n_positions * self.transformer.config.n_embd
        output_dim = len(self.label_mapping)
        self.linear_model = LinearModel(input_dim, output_dim)
        self.linear_model.to(self.device)
        optimizer = torch.optim.Adam(self.linear_model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        # Train
        for i in range(self.num_epochs):
            print("Running epoch {}/{}...".format(i + 1, self.num_epochs))
            pbar = tqdm(dataloader)
            for x, y in pbar:
                with torch.no_grad():
                    hidden, past = self.transformer(x)
                    hidden = hidden.view(len(hidden), -1)
                y_pred = self.linear_model(hidden)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.set_description("Loss: {:8.4f}".format(loss))

        # Save model
        torch.save(self.linear_model, self.model_save_path)

    def test(self, config):
        self.init_params(config, setup_mode='test')
        self.load_models()

        # Load test data
        test_data = MyDataset(config.test_data, transform=Encoder(self.tokenizer, self.label_mapping, device=self.device, max_size=self.transformer.config.n_positions))
        dataloader = DataLoader(test_data, batch_size=self.test_batch_size)

        true_labels = torch.empty(0, dtype=torch.int64, device=self.device)
        predicted_labels = torch.empty(0, dtype=torch.int64, device=self.device)
        for x, y in tqdm(dataloader):
            pred = self.predict_from_tensor(x)
            true_labels = torch.cat((true_labels, y))
            predicted_labels = torch.cat((predicted_labels, pred))

        metrics = self.performance_metrics(true_labels.cpu(), predicted_labels.cpu(), label_mapping=self.get_label_mapping(config))
        return metrics

    def predict(self, config, data):
        self.init_params(config, setup_mode='predict')
        self.load_models()

        # write to disk
        f_path = os.path.join('.', 'tmp', 'predict.csv')
        pd.DataFrame({'text': data}).to_csv(f_path)

        # Load test data
        predict_data = MyDatasetPredict(f_path, transform=EncoderPredict(self.tokenizer, self.label_mapping, device=self.device, max_size=self.transformer.config.n_positions))
        dataloader = DataLoader(predict_data, batch_size=self.test_batch_size)

        predictions = []
        for x in tqdm(dataloader):
            preds = self.predict_from_tensor(x, max_value=False)
            preds = preds.cpu().numpy()
            res = self.format_predictions(preds, label_mapping=self.label_mapping)
            predictions.extend(res)
        return predictions 

    def predict_from_tensor(self, x, max_value=True):
        with torch.no_grad():
            hidden, past = self.transformer(x)
            hidden = hidden.view(len(hidden), -1)
            pred = self.linear_model(hidden)
            if max_value:
                return torch.argmax(pred, 1)
            else:
                return torch.nn.functional.softmax(pred, dim=1)

    def load_models(self, transformer_only=False):
        """Loads the transformer and the linear model
        :param transformer_only: Load only transformer (default: False)
        """
        # Tokenizer
        model_type = 'gpt2'
        if self.tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=self.model_path)
        # Transformer
        if self.transformer is None:
            self.transformer = GPT2Model.from_pretrained(model_type, cache_dir=self.model_path)
            self.transformer.eval()
            self.transformer.to(self.device)
        if not transformer_only and self.linear_model is None:
            self.linear_model = torch.load(self.model_save_path)
            self.linear_model.eval()
            self.linear_model.to(self.device)

    def load_lm(self):
        """Loads the GPT-2 language model"""
        model_type = 'gpt2'
        if self.tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_type, cache_dir=self.model_path)
        if self.transformer_lm is None:
            self.transformer_lm = GPT2LMHeadModel.from_pretrained(model_type, cache_dir=self.model_path)
            self.transformer_lm.to(self.device)
            self.transformer_lm.eval()

    def generate_text(self, seed_text='This is an example text'):
        if self.tokenizer is None or self.transformer_lm is None:
            self.init_params({}, setup_mode='generate_text')
            self.load_lm()
        context_tokens = self.tokenizer.encode(seed_text)
        out = self.sample_sequence(context_tokens, top_k=0)
        out = out[:, len(context_tokens):].tolist()
        text = self.tokenizer.decode(out[0])
        return text

    def top_k_logits(self, logits, k):
        if k == 0:
            return logits
        values, _ = torch.topk(logits, k)
        min_values = values[:, -1]
        return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)

    def sample_sequence(self, context, top_k=0):
        length = self.transformer_lm.config.n_ctx // 2
        context = torch.tensor(context, device=self.device, dtype=torch.long).unsqueeze(0).repeat(self.batch_size, 1)
        prev = context
        output = context
        past = None
        with torch.no_grad():
            for i in range(length):
                logits, past = self.transformer_lm(prev, past=past)
                logits = logits[:, -1, :] / self.temperature
                logits = self.top_k_logits(logits, k=top_k)
                log_probs = torch.nn.functional.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1)
                output = torch.cat((output, prev), dim=1)
        return output

    def update_seed(self):
        self.seed = np.random.randint(1e6)
        torch.random.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)


class LinearModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        return self.linear(x)

class MyDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path, engine='python')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = self.df.iloc[i][['text', 'label']].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

class MyDatasetPredict(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path, engine='python')
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        sample = self.df.iloc[i][['text']].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

class EncoderPredict:
    def __init__(self, tokenizer, label_mapping, device=None, max_size=768):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.device = device
        self.max_size = max_size

    def __call__(self, sample):
        text_encoded = torch.zeros(self.max_size, dtype=torch.int64, device=self.device)
        tokenized = self.tokenizer.encode(sample['text'])
        text_encoded[:len(tokenized)] = torch.tensor(tokenized, dtype=torch.int64, device=self.device)
        return text_encoded

    def make_one_hot(self, label):
        t = torch.eye(len(self.label_mapping))
        return t[self.label_mapping[label]]

class Encoder:
    def __init__(self, tokenizer, label_mapping, device=None, max_size=768):
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.device = device
        self.max_size = max_size

    def __call__(self, sample):
        text_encoded = torch.zeros(self.max_size, dtype=torch.int64, device=self.device)
        tokenized = self.tokenizer.encode(sample['text'])
        text_encoded[:len(tokenized)] = torch.tensor(tokenized, dtype=torch.int64, device=self.device)
        label = torch.tensor(self.label_mapping[str(sample['label'])], dtype=torch.int64).to(self.device)
        return text_encoded, label

    def make_one_hot(self, label):
        t = torch.eye(len(self.label_mapping))
        return t[self.label_mapping[label]]
