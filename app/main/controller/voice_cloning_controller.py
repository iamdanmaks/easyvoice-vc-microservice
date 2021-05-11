from flask import request
from flask_restplus import Resource

from flask_restplus import Namespace, fields

from ..service.voice_cloning_service import clone_voice


api = Namespace('voice', description='voice related operations')
voice = api.model('voice', {
        'public_id': fields.String(required=True, description='voice public id'),
        'file': fields.String(required=True, description='base64 string with encoded wav file')
    })


@api.route('/')
class VoiceCloning(Resource):
    @api.response(201, 'Voice successfully cloned.')
    @api.doc('clone a new voice')
    @api.expect(voice)
    def post(self):
        """Creates a new User """
        data = request.json
        return clone_voice(data.get('public_id'), data.get('file'))
