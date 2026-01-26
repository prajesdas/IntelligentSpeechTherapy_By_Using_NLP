# from dotenv import load_dotenv
# from flask import Flask
# from flask_cors import CORS
# import os
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_jwt_extended import JWTManager


# load_dotenv()

# app = Flask(__name__)

# POSTGRESQL_URI = os.getenv("POSTGRESQL_URI")
# SECRET_KEY = os.getenv("SECRET_KEY")

# app.config['SQLALCHEMY_DATABASE_URI'] = POSTGRESQL_URI
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")
# app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY") 

# CORS(app)
# jwt = JWTManager()
# db = SQLAlchemy()
# bcrypt = Bcrypt(app)


import os
from dotenv import load_dotenv

load_dotenv()
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv("POSTGRESQL_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    SECRET_KEY = os.getenv("SECRET_KEY")
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
