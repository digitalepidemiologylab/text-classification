import os
import logging
from utils.helpers import find_project_root
from utils.docker import Docker
from utils.ecr import ECR
from utils.s3 import S3
from utils.sagemaker import Sagemaker

logger = logging.getLogger(__name__)


def build(run, project, model_type):
    docker = Docker()
    docker_path = os.path.join(find_project_root(), 'sagemaker', model_type)
    image_name = get_image_name(run, project)
    docker.build(docker_path, image_name)

def push(run, project, model_type):
    docker = Docker()
    s3 = S3()
    image_name = get_image_name(run, project)
    docker.push(image_name)
    s3.upload_model(run, image_name, model_type=model_type)

def build_and_push(run, project, model_type):
    build(run, project, model_type)
    push(run, project, model_type)

def run_local(run, project, model_type):
    # build image
    build(run, project, model_type)
    # run it
    docker = Docker()
    image_name = get_image_name(run, project)
    docker.run(image_name, run, model_type)

def create_model_and_configuration(run, project, question_tag, model_type, instance_type):
    # init helpers
    ecr = ECR()
    s3 = S3()
    sm = Sagemaker()
    # build deploy arguments
    image_name = get_image_name(run, project)
    ecr_image_name = ecr.get_ecr_image_name(image_name)
    s3_model_path = s3.get_model_s3_path(image_name)
    tags = [{'Key': 'project_name', 'Value': project},
            {'Key': 'question_tag', 'Value': question_tag},
            {'Key': 'run_name', 'Value': run},
            {'Key': 'model_type', 'Value': model_type}]
    # create model and endpoint configuration
    sm.create_model_and_configuration(ecr_image_name, s3_model_path, tags=tags, instance_type=instance_type)

def deploy(run, project, question_tag, model_type, instance_type):
    # initialize stuff
    # build image and push to ECR
    build_and_push(run, project, model_type)
    # create model and endpoint configuration
    create_model_and_configuration(run, project, question_tag, model_type, instance_type)

def get_image_name(run, project):
    return f'crowdbreaks_{project}_{run}'
