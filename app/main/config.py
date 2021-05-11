import os


class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'my_precious_secret_key')
    DEBUG = False
    AWS_ACCESS_KEY_ID = 'AKIASC3HPVTHMV4CJGXK'
    AWS_SECRET_ACCESS_KEY = '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'
    BUCKET_NAME = 'diploma1'


class DevelopmentConfig(Config):
    # uncomment the line below to use postgres
    # SQLALCHEMY_DATABASE_URI = postgres_local_base
    DEBUG = True


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    PRESERVE_CONTEXT_ON_EXCEPTION = False


class ProductionConfig(Config):
    DEBUG = False


config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY
