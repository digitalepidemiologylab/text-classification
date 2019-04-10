from models.base_model import BaseModel
try:
    from fastText import train_supervised, load_model
except:
    pass
import csv
import os
from sklearn.metrics import accuracy_score, classification_report

class FastTextModel(BaseModel):
    model_name = 'fasttext'
    label_prefix = '__label__'
    classifier = None

    def __init__(self):
        super().__init__()
        self.label_mapping = None

    def train(self, config):
        train_data_path = self.generate_input_file(config.train_data, config.tmp_path)
        output_model_path = os.path.join(config.output_path, 'model.bin')
        self.label_mapping = self.set_label_mapping(config)
        print("Training FastText model...")
        self.classifier = train_supervised(
                input=train_data_path,
                lr=config.get('learning_rate', 0.1),
                dim=config.get('dimensions', 100),
                ws=5,
                epoch=config.get('num_epochs', 5),
                minCount=1,
                minCountLabel=0,
                minn=0,
                maxn=0,
                neg=5,
                wordNgrams=config.get('ngrams', 3),
                loss='softmax',
                bucket=2000000,
                thread=47,
                lrUpdateRate=100,
                t=config.get('learning_rate', 0.0001),
                label=self.label_prefix,
                verbose=2,
                pretrainedVectors='')
        self.classifier.save_model(output_model_path)

    def test(self, config):
        self.load_classifier(config)
        self.label_mapping = self.get_label_mapping(config)
        test_x, test_y = self.generate_input_file(config.test_data, config.tmp_path, in_memory=True)
        test_y = [self.label_mapping[y] for y in test_y]
        predictions = [self.label_mapping[p['labels'][0]] for p in self.predict(config, data=test_x)]
        return self.performance_metrics(test_y, predictions, label_mapping=self.get_label_mapping(config))

    def predict(self, config, data=None):
        self.load_classifier(config)
        candidates = self.classifier.predict(data, k=32)
        predictions = [{
            'labels': [label[len(self.label_prefix):] for label in candidate[0]],
            'probabilities': candidate[1].tolist()
        } for candidate in zip(candidates[0], candidates[1])]
        return predictions

    def generate_input_file(self, input_path, tmp_path, in_memory=False):
        if in_memory:
            X = []
            Y = []
        else:
            tmpfile_name = os.path.basename(input_path) + '.fasttext.tmp'
            tmpfile_path = os.path.join(tmp_path, tmpfile_name)
        with open(input_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            if in_memory:
                for row in reader:
                    X.append(row['text'])
                    Y.append(row['label'])
            else:
                with open(tmpfile_path, 'w') as datafile:
                    for row in reader:
                        datafile.write(' '.join([self.label_prefix + row['label'], row['text']]) + '\r\n')
        if in_memory:
            return X, Y
        return tmpfile_path

    def load_classifier(self, config):
        if self.classifier is None:
            output_model_path = os.path.join(config.output_path, 'model.bin')
            self.classifier = load_model(output_model_path)
