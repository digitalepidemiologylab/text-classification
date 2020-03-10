import os
import json
import pickle
import sys
import flask
import fasttext
from flask import jsonify, request
import re
import logging
from werkzeug.exceptions import HTTPException
import pickle
import time


# vars
model_path = '/opt/ml/model'
label_prefix = '__label__'

# logging
logger = logging.getLogger(__name__)

# flask app
app = flask.Flask(__name__)

class Classifier():
    model = None
    label_mapping = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = fasttext.load_model(os.path.join(model_path, 'model.bin'))
        return cls.model

    @classmethod
    def predict(cls, text):
        """For the input, do the predictions and return them."""
        clf = cls.get_model()
        label_mapping = cls.get_label_mapping()
        return clf.predict(text, k=len(label_mapping))

    @classmethod
    def get_label_mapping(cls):
        """Get label mapping (map of logits to labels)"""
        if cls.label_mapping is None:
            with open(os.path.join(model_path, 'label_mapping.pkl'), 'rb') as f:
                cls.label_mapping = pickle.load(f)
        return cls.label_mapping


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare it healthy if we can load the model successfully."""
    health = Classifier.get_model() is not None
    status = 200 if health else 404
    return jsonify(health=health, status=status), status

@app.route('/invocations', methods=['POST'])
def predict():
    time_start = time.time()
    content = request.get_json(silent=True)
    text = content['text']
    logger.info(f'Received text {text}')
    prediction = _predict(text)
    time_end = time.time()
    prediction['duration_ms'] = 1000*(time_end - time_start)
    return jsonify({'prediction': prediction}), 200

def _predict(text):
    text = sanitize(text)
    if text is None:
        return {}
    candidates = Classifier.predict(text)
    labels_fixed = get_labels_fixed_order()
    probabilities = candidates[1].tolist()
    labels = [label[len(label_prefix):] for label in candidates[0]]
    probabilities_fixed = [probabilities[labels.index(i)] for i in labels_fixed]
    return {
        'labels': labels,
        'probabilities': probabilities,
        'labels_fixed': labels_fixed,
        'probabilities_fixed': probabilities_fixed,
        'model_type': 'fasttext'
        }

def get_labels_fixed_order():
    label_mapping = Classifier.get_label_mapping()
    if label_mapping is None:
        return []
    else:
        return list(label_mapping.keys())

def sanitize(text, discard_word_length=2):
    """Sanitize input text"""
    # Replace unnecessary spacings/EOL chars
    try:
        text = text.replace('\n', '').replace('\r', '').strip()
    except:
        return None
    text = text.split()
    # throw away anything below certain words length
    if not discard_word_length < len(text) < 110:
        return None
    text = ' '.join(text)
    text = text.lower()
    # replace urls and mentions
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', text)
    text = re.sub('(\@[^\s]+)', '<user>', text)
    try:
        text = text.decode('unicode_escape').encode('ascii','ignore')
    except:
        pass
    filter(lambda word: ' ' not in word, text)
    return text.strip()
