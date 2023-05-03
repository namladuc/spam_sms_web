import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, abort
from flask import Response, json, jsonify, send_file, render_template_string
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import matplotlib.pyplot as plt
import os
import io
import MySQLdb.cursors
import datetime
import calendar
from calendar import monthrange
import pandas as pd
import numpy as np
import functools
import pdfkit
import re
import math

app = Flask(__name__)
app.secret_key = 'La Nam'

MODEL_FOLDER = 'static/model/'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'smsspamproject'

mysql = MySQL(app)

def login_required(func): # need for all router
    @functools.wraps(func)
    def secure_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.url))
        return func(*args, **kwargs)
    return secure_function

@app.route("/")
@app.route("/login", methods=['GET','POST'])
def login():
    if 'username' in session.keys():
        return redirect(url_for("home"))
    
    cur = mysql.connection.cursor()
    
    if request.method == 'POST':
        details = request.form
        user_name = details['username'].strip()
        password = hashlib.md5(details['current-password'].lower().encode()).hexdigest()
        
        cur.execute("""SELECT *
                FROM user us
                WHERE us.username = %s""",(user_name,))
        user_data = cur.fetchall()
        
        if len(user_data)==0:
            return render_template('general/login.html',
                                   user_exits='False',
                                   pass_check='False')
        
        if password != user_data[0][3]:
            return render_template('general/login.html',
                                   user_exits='True', 
                                   pass_check='False')
    
        my_user = user_data[0]
        
        cur.execute("""
                SELECT r.role_folder
                FROM role r
                JOIN role_user ru ON ru.id_role = r.role_id
                WHERE ru.id_user = %s
            """, (my_user[0], ))
        role = cur.fetchall()[0][0]
        session['role'] = role
        
        cur.execute("""
                SELECT r.role_id
                FROM role r
                JOIN role_user ru ON ru.id_role = r.role_id
                WHERE ru.id_user = %s
            """, (my_user[0], ))
        role_id = cur.fetchall()[0][0]
        session['role_id'] = role_id
        
        cur.close()
        return redirect(url_for("home"))
    return render_template('general/login.html')

@app.route("/register_account", methods=['GET','POST'])
def register_account():
    return render_template("general/register_account.html")

@app.route("/home", methods=['GET','POST'])
def home():
    return render_template(session['role'] + '/home.html')

@app.route("/form_add_model", methods=['GET','POST'])
def form_add_model():
    return render_template(session['role'] + "/form_add_model.html")