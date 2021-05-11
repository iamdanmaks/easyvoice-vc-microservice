import os
import unittest

from flask_script import Manager

from app.main import create_app

from app import blueprint


app = create_app(os.getenv('BOILERPLATE_ENV') or 'dev')
app.register_blueprint(blueprint)

app.app_context().push()

app.run()
