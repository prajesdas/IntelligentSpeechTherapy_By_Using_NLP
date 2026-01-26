from flask import Flask, app
from flask_cors import CORS

from config import Config
from extensions import db, jwt, bcrypt, migrate

from modules.auth.routes import auth_bp
from modules.assessments.routes import assess_bp
from modules.sentences.routes import sentences_bp
from modules.practice.routes import routes_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Extensions
    db.init_app(app)
    jwt.init_app(app)
    bcrypt.init_app(app)
    migrate.init_app(app, db)
    # from flask_cors import CORS

    # allow only the Vite origin and cookies
    FRONTEND_ORIGINS = "http://localhost:5173"

    CORS(
        app,
        resources={
            r"/auth/*": {"origins": FRONTEND_ORIGINS},
            r"/assessment/*": {"origins": FRONTEND_ORIGINS},
            r"/practice/*": {"origins": FRONTEND_ORIGINS},  # include the spelling your frontend uses
        },
        supports_credentials=True,
        expose_headers=["Content-Type", "Authorization"],
        allow_headers=["Content-Type", "Authorization"],
        methods=["GET", "POST", "OPTIONS"]
    )  # CORS(app)

    # Blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(assess_bp, url_prefix="/assessment")
    app.register_blueprint(sentences_bp, url_prefix="/sentences")
    app.register_blueprint(routes_bp, url_prefix="/practice")

    @app.route("/")
    def home():
        return "This is the home page."

    return app


if __name__ == "__main__":
    app = create_app()

    app.run(debug=True)
