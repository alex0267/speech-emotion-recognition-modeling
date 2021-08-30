import os

CONFIG_ENV = os.getenv("CONFIG", "dev")


class Config:
    DB_URL = ""
    GOOGLE_APPLICATION_CREDENTIALS_PATH = "./credentials.json"
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "test_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    PROJECT_ID = "wewyse-centralesupelec-ftv"

    @classmethod
    def db_url_string(cls):
        return f"postgresql+psycopg2://{cls.POSTGRES_USER}:{cls.POSTGRES_PASSWORD}@{cls.POSTGRES_HOST}:{cls.POSTGRES_PORT}/{cls.POSTGRES_DB}"


class DevConfig(Config):
    CONFIG_ENV = "dev"
    BUCKET_NAME = "swa-dev-bucket"


class TestConfig(Config):
    CONFIG_ENV = "test"
    BUCKET_NAME = "swa-test-bucket"


class ProdConfig(Config):
    CONFIG_ENV = "prod"


config = dict(dev=DevConfig, test=TestConfig, prod=ProdConfig)
