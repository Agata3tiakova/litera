import os

from dotenv import load_dotenv
from flask import Flask

from app.config import Config
from app.database import db


def create_app(config_object=Config):
    load_dotenv()

    app = Flask(__name__)
    app.config.from_object(config_object)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    with app.app_context():
        from app.routes import bp

        app.register_blueprint(bp)
        db.create_all()

    return app

__all__ = ["create_app", "db"]
