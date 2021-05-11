from flask_restplus import Api
from flask import Blueprint

from .main.controller.voice_cloning_controller import api as voice_ns


blueprint = Blueprint('api', __name__)

api = Api(blueprint,
            title='VOICE CLONING MICROSERVICE',
            version='1.0',
            description='flask restplus microservice'
          )

api.add_namespace(voice_ns, path='/voice')
