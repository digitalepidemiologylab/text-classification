import os
import logging
import boto3
import base64

logger = logging.getLogger(__name__)


class ECR():
    def __init__(self):
        self.boto_session = None

    @property
    def sess(self):
        if self.boto_session is None:
            self.boto_session = boto3.Session()
        return self.boto_session

    @property
    def region(self):
        return self.sess.region_name

    @property
    def client(self):
        return self.sess.client('ecr')

    @property
    def account(self):
        return self.sess.client('sts').get_caller_identity()['Account']

    @property
    def registry_url(self):
        return f'{self.account}.dkr.ecr.{self.region}.amazonaws.com'

    def get_auth_token(self):
        resp = self.client.get_authorization_token()
        token = resp['authorizationData'][0]['authorizationToken']
        token = base64.b64decode(token).decode()
        return token

    def list_repositories(self):
        repositories = self.client.describe_repositories()
        if 'repositories' in repositories:
            return [r['repositoryName'] for r in repositories['repositories']]
        return []

    def create_repository(self, name):
        if name not in self.list_repositories():
            logger.info(f'Creating ECR repository {name}...')
            tags = [{'Key': 'project', 'Value': 'crowdbreaks'}]
            self.client.create_repository(repositoryName=name, tags=tags)

    def get_ecr_image_name(self, image_name):
        ecr_image = f'{self.registry_url}/{image_name}:latest'
        return ecr_image
