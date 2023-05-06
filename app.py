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
import json

# plot
import plotly
import plotly.express as px
import plotly.figure_factory as ff

# import module
from data_model_process import *

app = Flask(__name__)
app.secret_key = 'La Nam'

MODEL_FOLDER = 'static/model/'
TFIDF_FOLDER = 'static/tfidf/'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'smsspamproject'

# Static folder
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['TFIDF_FOLDER'] = TFIDF_FOLDER

mysql = MySQL(app)

def login_required(func): # need for all router
    @functools.wraps(func)
    def secure_function(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login", next=request.url))
        return func(*args, **kwargs)
    return secure_function

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
@app.route("/login", methods=['GET','POST'])
def login():
    if 'role' in session.keys():
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
        session['username'] = my_user
         
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
    if request.method == 'POST':
        details = request.form
        user_name = details['username'].strip()
        full_name = details['fullname'].strip()
        role_id = 2
        password = details['password'].strip()
        confirm_password = details['confirm_password'].strip()

        if password != confirm_password:
            error = "Mật khẩu xác nhận không khớp!"
            return render_template('general/register_account.html', error=error)

        hashed_password = hashlib.md5(password.lower().encode()).hexdigest()

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO user (full_name, username, password) VALUES (%s, %s, %s)",
                    (full_name, user_name, hashed_password))

        user_id = cur.lastrowid  # Lấy id của user vừa thêm

        cur.execute("INSERT INTO role_user (id_role, id_user) VALUES (%s, %s)",
                    (role_id, user_id))

        mysql.connection.commit()
        cur.close()

        message = "Tài khoản đã được tạo thành công!"
        return render_template('general/register_account.html', message=message)

    return render_template('general/register_account.html')


@app.route("/home", methods=['GET','POST'])
def home():
    return render_template(session['role'] + '/home.html')

@app.route("/form_add_data_group_info", methods=['GET','POST'])
def form_add_data_group_info():
    cur = mysql.connection.cursor()
    
    if request.method == 'POST':
        details = request.form
        group_name = details['group_name']
        test_size = details['test_size']
        
        if 'FileTFIDFUpload' not in request.files.keys():
            return "Error 1"
        else:
            tfidf_file = request.files['FileTFIDFUpload']
            
        sql = """
            SELECT MAX(id_dgroup)
            FROM data_group_info
        """
        cur.execute(sql)
        id_max = cur.fetchall()
        if (id_max[0][0] == None):
            id_max = 1
        else:
            id_max = int(id_max[0][0]) + 1
            
        # file processing
        if tfidf_file.filename != '':
            if tfidf_file.filename.split(".")[-1] != 'pickle':
                return "Error"
            pathToFile = app.config['TFIDF_FOLDER'] + "tfidf_" + str(id_max) + ".pickle"
            tfidf_file.save(pathToFile)
            
        sql = """
            INSERT INTO data_group_info(group_name,
            tfidf_path, test_size) VALUES (%s, %s, %s)
        """
        cur.execute(sql, (group_name, pathToFile, test_size))
        mysql.connection.commit()
    return render_template(session['role'] + "/form_add_data_group_info.html")

@app.route("/form_add_model", methods=['GET','POST'])
def form_add_model():
    cur = mysql.connection.cursor()
    
    sql = """
        SELECT *
        FROM model_info
    """
    cur.execute(sql)
    model_infos = cur.fetchall()
    model_info_choice = []
    for elm in model_infos:
        tmp = " - ".join(list(elm[1:]))
        model_info_choice.append(tmp)
        
    sql = """
        SELECT id_dgroup, group_name
        FROM data_group_info
    """
    cur.execute(sql)
    data_group = cur.fetchall()
    
    if request.method == 'POST':
        details = request.form
        if 'FileModelUpload' not in request.files.keys():
            return "Error 1"
        else:
            model_file = request.files['FileModelUpload']
        time_train = details['time_train']
        can_use = 1 if 'can_use' in details.keys() else 0
        
        # find id
        sql = """
            SELECT MAX(id_train)
            FROM model_train_state
        """
        cur.execute(sql)
        id_max = cur.fetchall()
        if (id_max[0][0] == None):
            id_max = 1
        else:
            id_max = id_max[0][0] + 1
        
        # file processing
        if model_file.filename != '':
            if model_file.filename.split(".")[-1] != 'pickle':
                return "Error 2"
            pathToFile = app.config['MODEL_FOLDER'] + "/model_" + str(id_max) + ".pickle"
            model_file.save(pathToFile)
            
        sql = """
            INSERT INTO model_train_state(id_train, id_dgroup,
            path_to_state, can_use, time_train,
            create_by, update_by)  VALUES 
            (%s, %s, %s, %s, %s, %s, %s)
        """
            
        return redirect(url_for("home"))
    
    return render_template(session['role'] + "/form_add_model.html",
                           model_infos = model_info_choice,
                           data_group = data_group)
    
    
# ---- ADMIN ONLY ----
@app.route("/view_data_train", methods=['GET','POST'])
def view_data_train():
    # View du lieu trong bang data train
    cur = mysql.connection.cursor()
    sql = """
                SELECT *
                FROM data_train 
                join class on data_train.class_id = class.class_id
            """
    cur.execute(sql)
    dtrain = cur.fetchall()

    return render_template(session['role'] + "/view_data_train.html", data=dtrain)


@app.route("/view_data_group_info")
def view_data_group_info():
    cur = mysql.connection.cursor()
    sql = """
            SELECT *
            FROM data_group_info
        """
    cur.execute(sql)
    dgroup_infos = cur.fetchall()

    return render_template(session['role'] + "/view_data_group_info.html", data=dgroup_infos)

@app.route("/view_one_group_info/<string:id_dgroup>", methods=['GET','POST'])
def view_one_group_info(id_dgroup):
    # pie chart: %ham, %spam
    # bar plot: word frequency for ham and spam label
    # table for statistic model
    # word cloud
    # M: Modify
    # R: Read
    
    cur = mysql.connection.cursor()
    
    # check exist group id
    cur.execute("""
                SELECT id_dgroup, group_name
                FROM data_group_info
                WHERE id_dgroup = %s
                """, (id_dgroup, ))
    check_exist = cur.fetchall()
    if len(check_exist) == 0:
        return "Error"
    
    sql = """
        SELECT dt.text, dt.class_id, c.label
        FROM data_train dt 
        JOIN class c ON dt.class_id = c.class_id
        WHERE dt.id_dtrain IN (
            SELECT dgs.id_dtrain
            FROM data_group_split dgs
            WHERE dgs.id_dgroup = %s
        )
    """
    cur.execute(sql, (id_dgroup, ))
    records = cur.fetchall()
    if len(records) == 0:
        return "Error"
    
    columnName = ['Text', 'Class', 'Label']
    data = pd.DataFrame.from_records(records, columns=columnName)
    
    # Metadata
    num_rows = data.shape[0]
    
    # for pie chart
    df2 = data.apply(lambda x : True
            if x['Class'] == 1 else False, axis = 1)
    count_spam = len(df2[df2 == True].index)
    count_ham = len(df2[df2 == False].index)
    df2 = pd.DataFrame({
        "Label": ["Spam","Ham"],
        "Count": [count_spam, count_ham]
    })
    
    fig_pie = px.pie(df2, values='Count', names='Label', title='Percent of Spam and Ham ' + check_exist[0][1])
    graphJSON_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    del df2
    del count_ham
    del count_spam
    del fig_pie
    
    # for distribution of length spam
    colors = ['slategray', 'magenta']
    text_words_spam = data[data["Class"] == 1]["Text"].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)) # spam
    text_words_ham = data[data["Class"] == 0]["Text"].str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)) # ham
    fig_spam_distribution = ff.create_distplot([text_words_spam, text_words_ham],
                                               ['Spam','Ham'],
                                               curve_type='normal',
                                               colors=colors)
    fig_spam_distribution.update_layout(title_text="Distribution of average word length in texts where target is 'spam' and 'ham'")
    graphJSON_distribution = json.dumps(fig_spam_distribution, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(session['role'] + "/view_one_group_info.html",
                           graphJSON_pie = graphJSON_pie,
                           graphJSON_distribution = graphJSON_distribution,
                           num_rows = num_rows,
                           group_info = check_exist[0])

@app.route("/view_model_info", methods=['GET','POST'])
def view_model_info():
    return None

@app.route("/view_model_train_state", methods=['GET','POST'])
def view_model_train_state():
    return None

@app.route("/view_one_model_train_state/<string:id_train>_<string:mode>", methods=['GET','POST'])
def view_one_model_train_state(id_train, mode):
    return None

# @app.route("/view_data_group/<string:group_id>_<string:mode>", methods=['GET','POST'])
# def view_data_group(group_id, mode):
#     return None

@app.route("/check_data_input", methods=['GET','POST'])
def check_data_input():
    return None

@app.route("/view_account", methods=['GET','POST'])
def view_account():
    cur = mysql.connection.cursor()
    sql = """
                SELECT * FROM user
                join role_user on user.id = role_user.id_user
                join role on role.role_id = role_user.id_role
          """
    cur.execute(sql)
    accounts = cur.fetchall()

    return render_template(session['role'] + "/view_account.html", data=accounts)

@app.route("/form_add_account", methods=['GET','POST'])
def form_add_account():
    cur = mysql.connection.cursor()
    sql = """
            SELECT *
            FROM role
          """
    cur.execute(sql)
    list_role = cur.fetchall()

    if request.method == 'POST':
        details = request.form
        user_name = details['username'].strip()
        full_name = details['fullname'].strip()
        role_id = details['role'].strip()
        password = details['password'].strip()
        confirm_password = details['confirm_password'].strip()

        if password != confirm_password:
            error = "Mật khẩu xác nhận không khớp!"
            return render_template(session['role'] + '/form_add_account.html', error=error,  list_role=list_role)

        hashed_password = hashlib.md5(password.lower().encode()).hexdigest()

        cur.execute("INSERT INTO user (full_name, username, password) VALUES (%s, %s, %s)",
                    (full_name, user_name, hashed_password))

        user_id = cur.lastrowid   # Lấy id của user vừa thêm

        cur.execute("INSERT INTO role_user (id_role, id_user) VALUES (%s, %s)",
                    (role_id, user_id))

        mysql.connection.commit()
        cur.close()

        message = "Tài khoản đã được tạo thành công!"
        return render_template(session['role'] + '/form_add_account.html', message=message,  list_role=list_role)

    return render_template(session['role'] + '/form_add_account.html', list_role=list_role)


@app.route("/form_add_data_train", methods=['GET','POST'])
def form_add_data_train():
    cur = mysql.connection.cursor()
    
    if request.method == 'POST':
        if 'FileDataTrainUpload' not in request.files.keys():
            return "Error 1"
        else:
            data_file = request.files['FileDataTrainUpload']
            
        # file processing
        if data_file.filename == '':
            return "Error"
        
        if data_file.filename.split(".")[-1] != 'csv':
            return "Error"
        
        data = pd.read_csv(data_file, index_col=0, delimiter="|")
        
        # template sql 
        sql_data_train_template = "INSERT INTO data_train(text, class_id, create_by, update_by) VALUES (%s, %s, %s, %s)"
        sql_select_data_train_id = "SELECT id_dtrain FROM data_train WHERE text = %s"
        # insert group
        sql_check_group = "SELECT * FROM data_group_split WHERE id_dtrain = %s AND id_dgroup = %s"
        sql_insert_group = "INSERT INTO data_group_split(id_dtrain, id_dgroup) VALUES (%s, %s)"
        
        data['group_ids'] = data['group_ids'].astype(str)
        
        if len(list(set(data['group_ids'].values))) != 1:
            id_exists = list(set(",".join(list(data['group_ids'].values)).split(",")))
        else:
            id_exists = list(set(data['group_ids'].values))
            
        for elm in id_exists:
            # check group_id exist
            cur.execute("""
                        SELECT *
                        FROM data_group_info
                        WHERE id_dgroup = %s
                        """, (int(elm), ))
            check_group_exist = cur.fetchall()
            if len(check_group_exist) == 0:
                id_exists.remove(elm)

        for i in range(data.shape[0]):
            label = data['Target'][i]
            text = str(data['corpus'][i])
            group_ids = data['group_ids'][i]
            
            if len(text) == 0:
                continue
            
            # check if text is in database
            # if exist -> insert + take id
            # else -> take id
            cur.execute(sql_select_data_train_id, (text, ))
            data_id = cur.fetchall()
            if len(data_id) == 0:
                cur.execute(sql_data_train_template, (text, label,
                                                      session['username'][1],
                                                      session['username'][1]))
                mysql.connection.commit()
                cur.execute(sql_select_data_train_id, (text, ))
                data_id = cur.fetchall()
            data_id = data_id[0][0]
            
            for id_group in group_ids.split(","):
                
                if id_group not in id_exists:
                    continue
                
                cur.execute(sql_check_group, (data_id, int(id_group)))
                check_group_exist = cur.fetchall()
                if len(check_group_exist) == 0:
                    cur.execute(sql_insert_group, (data_id, int(id_group)))
                    mysql.connection.commit()
        return redirect(url_for('form_add_data_train'))
    return render_template(session['role'] + "/form_add_data_train.html")

@app.route("/change_to_data_train/<string:id_data_input>", methods=['GET','POST'])
def change_to_data_train(id_data_input):
    return None

# ---- ADMIN ONLY ----