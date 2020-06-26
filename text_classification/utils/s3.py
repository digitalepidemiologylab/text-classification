import os
import boto3
import logging
import tarfile
from text_classification.utils.helpers import find_project_root


logger = logging.getLogger(__name__)

class S3():
    def __init__(self, bucket_name='crowdbreaks-sagemaker'):
        self.bucket_name = bucket_name
        self.s3_client = None
        self.tar_name = 'model.tar.gz'

    @property
    def client(self):
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def upload_model(self, run, image_name,  model_type='fasttext'):
        run_dir = os.path.join(find_project_root(), 'output', run)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f'Could not find run directory {run_dir}')
        # compile model artefacts
        default_model_files = ['label_mapping.pkl', 'run_config.json']
        if model_type == 'fasttext':
            model_files = ['model.bin']
        else:
            raise ValueError(f'Model type {model_type} is not yet supported')
        model_files += default_model_files
        input_files = [os.path.join(run_dir, model_file) for model_file in model_files]
        # tar
        model_tarfile = os.path.join(run_dir, self.tar_name)
        self.make_tar(input_files, model_tarfile)
        # upload
        key = self.get_model_s3_key(image_name)
        logger.info(f'Uploading {model_tarfile} to S3 bucket {self.bucket_name} under key {key}...')
        self.client.upload_file(model_tarfile, self.bucket_name, key)

    def get_model_s3_key(self, image_name):
        return os.path.join('output', image_name, self.tar_name)

    def get_model_s3_path(self, image_name):
        key = self.get_model_s3_key(image_name)
        return f's3://{self.bucket_name}/{key}'

    def make_tar(self, input_files, output_file):
        with tarfile.open(output_file, "w:gz") as tar:
            for input_file in input_files:
                logger.info(f'Adding {input_file} to tarfile {output_file}...')
                tar.add(input_file, arcname=os.path.basename(input_file))
