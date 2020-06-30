import boto3
import logging
from ..utils import Docker
from .ecr import ECR
import uuid

logger = logging.getLogger(__name__)


class Sagemaker():
    def __init__(self, role_name='crowdbreaks-sagemaker'):
        self.sm_client = None
        self.boto_session = None
        self.role_name = role_name

    @property
    def sess(self):
        if self.boto_session is None:
            self.boto_session = boto3.Session()
        return self.boto_session

    @property
    def account(self):
        return self.sess.client('sts').get_caller_identity()['Account']

    @property
    def role(self):
        return f'arn:aws:iam::{self.account}:role/{self.role_name}'

    @property
    def client(self):
        if self.sm_client is None:
            self.sm_client = self.sess.client('sagemaker')
        return self.sm_client

    def create_model(self, model_name, ecr_image_name, s3_model_key, tags):
        primary_container = {'Image': ecr_image_name, 'ModelDataUrl': s3_model_key}
        resp = self.client.create_model(
            ModelName = model_name,
            ExecutionRoleArn = self.role,
            PrimaryContainer = primary_container,
            Tags = tags)
        model_arn = resp['ModelArn']
        logger.info(f'Successfully created sagemaker model {model_arn}')

    def create_endpoint_configuration(self, model_name, instance_type):
        endpoint_config_name = f'{model_name}-config'
        tags = [{'Key': 'project', 'Value': 'crowdbreaks'}]
        resp = self.client.create_endpoint_config(
                EndpointConfigName = endpoint_config_name,
                ProductionVariants=[{
                    'InstanceType': instance_type,
                    'InitialVariantWeight': 1,
                    'InitialInstanceCount': 1,
                    'ModelName': model_name,
                    'VariantName':'AllTraffic'}],
                Tags=tags)
        endpoint_config_arn = resp['EndpointConfigArn']
        logger.info(f'Successfully created sagemaker endpoint config {endpoint_config_arn}')

    def create_model_and_configuration(self, ecr_image_name, s3_model_key, tags=[], instance_type='ml.t2.medium'):
        random_hash = uuid.uuid4().hex[:10]
        model_name = f'crowdbreaks-{random_hash}'
        logger.info(f'Creating model {model_name}')
        self.create_model(model_name, ecr_image_name, s3_model_key, tags)
        logger.info(f'Creating endpoint configuration for model {model_name} using a {instance_type} instance...')
        self.create_endpoint_configuration(model_name, instance_type)
