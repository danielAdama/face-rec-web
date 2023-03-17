from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import config
from dotenv import load_dotenv
import os
from flask_login import LoginManager


dbPath = os.path.join(os.getcwd(),"faceDb.sqlite3")
load_dotenv()

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config['FLASK_DEBUG'] = 1
    app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{dbPath}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
    
    from .models import User, Face
    from .account import account as auth_blueprint
    from .main import main as main_blueprint
    app.register_blueprint(auth_blueprint)
    app.register_blueprint(main_blueprint)
    db.init_app(app)

    if not os.path.exists(dbPath):
        with app.app_context():
            db.create_all()
            print('Database Created? ',os.path.exists(dbPath))
    
    login_manager = LoginManager()
    login_manager.login_view = 'account.login'
    login_manager.session_protection = "strong"
    login_manager.init_app(app)
    @login_manager.user_loader
    def load_user(user_id):
        # since the user_id is just the primary key of our user table, use it in the query for the user
        return User.query.get(int(user_id))

    return app