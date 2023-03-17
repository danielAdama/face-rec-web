from flask import Blueprint, Flask, render_template, Response, request, make_response, jsonify, redirect, url_for, flash
from . import db
from werkzeug.security import generate_password_hash, check_password_hash
from .models import User
from flask_login import login_user, login_required, logout_user
import datetime

account = Blueprint('account', __name__)

@account.route('/login')
def login():
    return render_template('login.html')

@account.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email').lower()
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False
    user = User.query.filter_by(email=email).first()
    # check if the user actually exists
    # take the user-supplied password, hash it, and compare it to the hashed password in the database
    if not user or not check_password_hash(user.password, password):
        flash('Please check your login details and try again.')
        return redirect(url_for('account.login')) # if the user doesn't exist or password is wrong, reload the page
    
    login_user(
        user, 
        remember = remember, 
        duration = datetime.timedelta(seconds=5)
    )
    return redirect(url_for('main.home'))

@account.route('/signup')
def signup():
    return render_template('signup.html')

@account.route('/signup', methods=['POST'])
def signup_post():
    if request.method == 'POST':
        email = request.form.get('email').lower()
        name = request.form.get('name')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists')
            return redirect(url_for('account.signup'))
        
        new_user = User(
            email=email, 
            name=name, 
            password=generate_password_hash(password, method='sha256')
        )
        db.session.add(new_user)
        db.session.commit()

    return redirect(url_for('account.login'))

@account.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('main.home'))