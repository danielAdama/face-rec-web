from . import db
from sqlalchemy.sql import func
from flask_login import UserMixin

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(100))
    def __repr__(self):
        return f'<User {self.id} {self.name}>'

class Face(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable = False)
    filename = db.Column(db.Text(100))
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    def __repr__(self):
        return f'<Face {self.id} {self.user_id} {self.filename} {self.created_at}>'