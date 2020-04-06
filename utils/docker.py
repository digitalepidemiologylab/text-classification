import os
import docker
import logging
from docker.errors import APIError, BuildError, ImageNotFound
from utils.helpers import find_project_root
from utils.ecr import ECR

logger = logging.getLogger(__name__)

class Docker():
    def __init__(self):
        self.docker_client = None
        self.docker_api_client = None
        self.ecr = ECR()

    @property
    def client(self):
        if self.docker_client is None:
            self.docker_client = docker.from_env()
        return self.docker_client

    @property
    def api_client(self):
        if self.docker_api_client is None:
            self.docker_api_client =  docker.APIClient()
        return self.docker_api_client

    def get_image(self, image_name):
        try:
            image = self.client.images.get(image_name)
        except ImageNotFound as e:
            logger.error(f'Could not find image {image_name}')
            raise e
        return image

    def build(self, docker_path, image_name):
        # build local image
        labels = {'platform': 'crowdbreaks'}
        logger.info(f'Building Docker image from {docker_path}...')
        try:
            logs = self.api_client.build(path=docker_path, tag=image_name, rm=True, labels=labels, decode=True, nocache=False)
            self._get_stream_log(logs)
        except (APIError, BuildError, TypeError) as e:
            logger.error('Build unsucessful. Build log:')
            self._get_stream_log(logs)
            raise e

    def run(self, image_name, run, model_type):
        """Serves model locally"""
        model_path = os.path.join(find_project_root(), 'output', run)
        docker_path = os.path.join(find_project_root(), 'sagemaker', model_type, 'src')
        # bind model artefacts to /opt/ml and bind a volume for the code
        volumes = {model_path: {'bind': '/opt/ml/model', 'mode': 'ro'}, docker_path: {'bind': '/opt/program'}}
        ports = {'5000':'5000'}
        command = 'python3 serve'
        environment = {'FLASK_ENV': 'DEVELOPMENT'}
        try:
            container = self.client.containers.run(image_name, command, volumes=volumes, ports=ports, detach=True, environment=environment)
            logs = container.logs(stream=True)
            self._get_stream_log(logs, decode=True)
        except KeyboardInterrupt:
            logger.info(f'Gracefully stopping docker container {image_name}...')
            container.stop()

    def list_tags(self):
        resp = self.client.images.list(filters={'label': 'platform=crowdbreaks'})
        tags = []
        for r in resp:
            tags.extend(r.attrs['RepoTags'])
        return tags

    def image_exists(self, image_name):
        try:
            self.client.images.get(image_name)
        except:
            return False
        else:
            return True

    def get_ecr_auth_config(self):
        token = self.ecr.get_auth_token()
        username, password = token.split(':')
        return {'username': username, 'password': password}

    def push(self, image_name):
        # check if local image exists
        if not self.image_exists(image_name):
            logger.error(f'Image {image_name} does not exist locally. You need to build it first!')
            return
        # login to AWS ECR
        logger.info('Attempting to log in to AWS ECR...')
        auth_config = self.get_ecr_auth_config()
        # create repository if it doesn't exist yet
        self.ecr.create_repository(image_name)
        # tag existing image with ECR name
        ecr_name = self.ecr.get_ecr_image_name(image_name)
        logger.info(f'Tagging image {image_name} as {ecr_name}...')
        image = self.get_image(image_name)
        image.tag(ecr_name)
        logger.info('Pushing tag to AWS ECR...')
        try:
            logs = self.api_client.push(ecr_name, stream=True, decode=True, auth_config=auth_config)
            self._get_stream_log(logs)
        except APIError as e:
            logger.error('Pushing image to ECR unsucessful.')
            self._get_stream_log(logs)
            raise e
        else:
            logger.info('Successfully pushed image to ECR.')


    # private

    def _get_stream_log(self, stream_log, decode=False):
        for log in stream_log:
            if decode:
                log = log.decode().strip()
            if 'stream' in log:
                out_log = log['stream'].strip()
                if len(out_log) > 0:
                    logger.info(out_log)
            elif 'errorDetail' in log:
                raise Exception(log['errorDetail'])
            else:
                logger.info(log)

