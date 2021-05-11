import boto3

from flask import Flask

from app.main.util.inference import load_model
from .config import config_by_name, Config


model = load_model()
client = boto3.client(
        's3',
        aws_access_key_id=Config.AWS_ACCESS_KEY_ID, 
        aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY
    )

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])

    return app
