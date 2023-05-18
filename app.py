import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, abort, flash
from flask import Response, json, jsonify, send_file, render_template_string
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from urllib.parse import quote
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
from datetime import datetime

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
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        details = request.form
        user_name = details['username'].strip()
        full_name = details['fullname'].strip()
        role_id = 2
        password = details['password'].strip()
        confirm_password = details['confirm_password'].strip()

        cur.execute("SELECT username FROM user")
        existing_usernames = [row[0] for row in cur.fetchall()]
        if user_name in existing_usernames:
            error = "Tên người dùng đã tồn tại trong hệ thống!"
            return render_template('general/register_account.html', error=error)

        if password != confirm_password:
            error = "Mật khẩu xác nhận không khớp!"
            return render_template('general/register_account.html', error=error)

        hashed_password = hashlib.md5(password.lower().encode()).hexdigest()

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
    if session['role_id'] != 1:
        return render_template(session['role'] + '/home.html', userinfo=session['username'])

    cur = mysql.connection.cursor()
    
    # info about num of group data
    cur.execute("""
                SELECT COUNT(*)
                FROM data_group_info
                """)
    num_of_group_data = cur.fetchall()[0][0]
    
    # info about train data
    cur.execute("""
                SELECT COUNT(*)
                FROM data_train
                """)
    count_train_data = cur.fetchall()[0][0]
    
    # info input data
    cur.execute("""
                SELECT COUNT(*)
                FROM data_input
                """)
    count_data_input = cur.fetchall()[0][0]
    
    # info num of model 
    cur.execute("""
                SELECT COUNT(*)
                FROM model_train_state
                """)
    count_num_model = cur.fetchall()[0][0]
    
    # info num of model active
    cur.execute("""
                SELECT COUNT(*)
                FROM model_train_state
                WHERE can_use = 1
                """)
    count_num_model_active = cur.fetchall()[0][0]
    
    # plot for data group count elm
    cur.execute("""
                SELECT dgi.group_name, COUNT(dgs.id_dtrain)
                FROM data_group_info dgi
                JOIN data_group_split dgs ON dgi.id_dgroup = dgs.id_dgroup
                GROUP BY dgi.id_dgroup
                ORDER BY COUNT(dgs.id_dtrain) DESC
                LIMIT 5
                """)
    records = cur.fetchall()
    columnName = ['group_name', 'count']
    data_models = pd.DataFrame.from_records(records, columns=columnName)
    fig_group_count = px.bar(data_models,
                    y='group_name',
                    x='count',
                    color='group_name',
                    text='count',
                    title="Top 5 large group dataset")
    fig_group_count.update_layout(xaxis_title="Count records",
                                yaxis_title="Group name",
                                legend_title="Group name")
    graphJSON_group_count = json.dumps(fig_group_count, cls=plotly.utils.PlotlyJSONEncoder)
    
    # num check spam in each day
    cur.execute("""
                SELECT DATE(di.create_at), COUNT(di.id) 
                FROM data_input di 
                GROUP BY DATE(di.create_at)
                ORDER BY DATE(di.create_at) ASC
                """)
    records = cur.fetchall()
    columnName = ['day', 'count']
    data_days = pd.DataFrame.from_records(records, columns=columnName)
    fig_day_input = px.area(data_days,
                            x='day',
                            y='count',
                            title="Spam input to check in each day")
    fig_day_input.update_layout(xaxis_title="Date",
                                yaxis_title="Count sms")
    graphJSON_day_input = json.dumps(fig_day_input, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    return render_template(session['role'] + '/home.html',
                           num_of_group_data = num_of_group_data,
                           count_train_data = count_train_data,
                           count_data_input = count_data_input,
                           count_num_model = count_num_model,
                           count_num_model_active = count_num_model_active,
                           graphJSON_group_count = graphJSON_group_count,
                           graphJSON_day_input = graphJSON_day_input,
                           userinfo = session['username'])

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
    return render_template(session['role'] + "/form_add_data_group_info.html", userinfo = session['username'])

@app.route("/view_model_info/view_model_train_state/form_add_model", methods=['GET','POST'])
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
            pathToFile = app.config['MODEL_FOLDER'] + "model_" + str(id_max) + ".pickle"
            model_file.save(pathToFile)
            
        # take id_dgroup
        id_data_choice = details['info_data']
        for elm in data_group:
            if id_data_choice == elm[1]:
                id_data_choice = elm[0]
                break
            
        # accuracy model
        acc_model_train = details['acc_model_train']
        acc_model_test = details['acc_model_test']
            
        sql = """
            INSERT INTO model_train_state(id_train, id_dgroup,
            path_to_state, can_use, time_train, accuracy_model_train, accuracy_model_test,
            create_by, update_by)  VALUES 
            (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cur.execute(sql, (id_max, id_data_choice, pathToFile,
                          can_use, time_train, acc_model_train, acc_model_test, session['username'][1],
                          session['username'][1]))
        mysql.connection.commit()
        
        # model train table 
        model_info = details['model_info']
        for i in range(len(model_info_choice)):
            if model_info == model_info_choice[i]:
                model_info = model_infos[i][0]
        
        cur.execute("""
                    INSERT INTO model_train(id_train, id_model)
                    VALUES (%s, %s)
                    """, (id_max, model_info))
        mysql.connection.commit()
            
        return redirect(url_for("home"))
    
    return render_template(session['role'] + "/form_add_model.html",
                           model_infos = model_info_choice,
                           data_group = data_group,
                           userinfo = session['username'])
    

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

    return render_template(session['role'] + "/view_data_train.html", data=dtrain, userinfo = session['username'])


@app.route("/update_one_data_train/<int:id_dtrain>", methods=['GET', 'POST'])
def update_one_data_train(id_dtrain):
    cur = mysql.connection.cursor()
    sql = """
                    select * 
                    from data_train
                    join data_group_split on data_train.id_dtrain = data_group_split.id_dtrain
                    WHERE data_train.id_dtrain = %s
                  """
    cur.execute(sql, (id_dtrain,))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"

    info_one_data_train = records[0]

    id_dgroup_list_selected = []
    for record in records:
        id_dgroup_list_selected.append(record[8])

    sql_group_name = """
                        select * 
                        from data_group_info
                        """
    cur.execute(sql_group_name)
    group_name_list = cur.fetchall()

    if request.method == 'POST':
        details = request.form
        data_clean = step_corpus_for_one_text(details['text_original'].strip())
        class_id = details['class_id'].strip()

        cur.execute("SELECT * FROM data_train WHERE text = %s", (data_clean,))
        check = cur.fetchall()
        if (len(check) != 0) and id_dtrain not in [item[0] for item in check]:
            return "Error2"

        lst_group = list(details.keys())
        lst_group.remove('text_original')
        lst_group.remove('class_id')

        # template sql
        # sql_select_data_train_id = "SELECT id_dtrain FROM data_train WHERE text = %s"
        #
        # cur.execute(sql_select_data_train_id, (data_clean,))
        # data_id = cur.fetchall()
        # if len(data_id) == 0 and id_dtrain not in [item[0] for item in data_id]:
        now = datetime.now()
        update_at = now.strftime("%Y-%m-%d %H:%M:%S")

        cur.execute("UPDATE data_train SET text = %s, class_id = %s, update_at = %s, update_by = %s WHERE id_dtrain = %s",
                    (data_clean, class_id, update_at, session['username'][1], id_dtrain))
        mysql.connection.commit()

        cur.execute("""
                            DELETE FROM data_group_split WHERE id_dtrain = %s;
                            """, (id_dtrain, ))
        mysql.connection.commit()

        for elm in group_name_list:
            if elm[1] in lst_group:
                cur.execute("""
                                    INSERT INTO data_group_split(id_dtrain, id_dgroup)
                                    VALUES (%s, %s)
                                    """, (id_dtrain, elm[0]))
                mysql.connection.commit()

        cur.close()

        flash("Chỉnh sửa thành công !!!")
        return redirect(url_for('view_data_train'))

    return render_template(session['role'] + "/update_one_data_train.html",
                           data=info_one_data_train,
                           id_dgroup_list_selected=id_dgroup_list_selected,
                           group_name_list=group_name_list, userinfo = session['username'])

@app.route("/delete_one_data_train/<int:id_dtrain>")
def delete_one_data_train(id_dtrain):
    cur = mysql.connection.cursor()
    sql1 = """
            SELECT * FROM data_train WHERE data_train.id_dtrain = %s
        """
    cur.execute(sql1, (id_dtrain,))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"
    sql2 = """
                    SELECT * FROM data_group_split WHERE data_group_split.id_dtrain = %s
                """
    cur.execute(sql2, (id_dtrain,))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"

    sql_data_train = """
                    DELETE FROM data_train
                    WHERE id_dtrain = %s;
                  """
    sql_data_group_split = """
                        DELETE FROM data_group_split
                        WHERE id_dtrain = %s;
                      """

    cur.execute(sql_data_group_split, (id_dtrain,))
    cur.execute(sql_data_train, (id_dtrain,))

    mysql.connection.commit()

    flash("Đã xóa thành công bản ghi dữ liệu huấn luyện có ID: " + str(id_dtrain))
    cur.close()
    return redirect(url_for("view_data_train"))


@app.route("/view_data_group_info")
def view_data_group_info():
    cur = mysql.connection.cursor()
    sql = """
            SELECT *
            FROM data_group_info
        """
    cur.execute(sql)
    dgroup_infos = cur.fetchall()

    return render_template(session['role'] + "/view_data_group_info.html", data=dgroup_infos, userinfo=session['username'])

@app.route("/update_one_group_info/<int:id_dgroup>", methods=['GET','POST'])
def update_one_group_info(id_dgroup):
    cur = mysql.connection.cursor()
    sql = """
                SELECT * 
                from data_group_info
                WHERE data_group_info.id_dgroup = %s;
              """
    cur.execute(sql, (id_dgroup,))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"

    record = records[0]

    if request.method == 'POST':
        details = request.form
        group_name = details['group_name'].strip()
        test_size = details['test_size'].strip()

        cur.execute("UPDATE data_group_info SET group_name = %s, test_size = %s WHERE id_dgroup = %s",
                    (group_name, test_size, id_dgroup))
        mysql.connection.commit()
        cur.close()

        flash("Chỉnh sửa thành công !!!")
        return redirect(url_for('view_data_group_info'))
    return render_template(session['role'] + "/update_one_group_info.html", data=record, userinfo=session['username'])

@app.route("/delete_one_group_info/<int:id_dgroup>")
def delete_one_group_info(id_dgroup):
    cur = mysql.connection.cursor()
    sql = """
            SELECT * 
            from data_group_split
            WHERE data_group_split.id_dgroup = %s;
          """
    cur.execute(sql, (id_dgroup,))
    records = cur.fetchall()

    if len(records) != 0:
        flash("Không thể xóa nhóm dữ liệu này!!")
    else:
        sql = """
                DELETE FROM data_group_info
                WHERE id_dgroup = %s;
              """
        cur.execute(sql, (id_dgroup,))
        mysql.connection.commit()
        flash("Đã xóa thành công nhóm dữ liệu có ID: " + str(id_dgroup))
    cur.close()
    return redirect(url_for("view_data_group_info"))

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
    text_words_spam = data[data["Class"] == 1]["Text"].apply(len) #str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)) # spam
    text_words_ham = data[data["Class"] == 0]["Text"].apply(len) #str.split().apply(lambda x : [len(i) for i in x]).map(lambda x: np.mean(x)) # ham
    fig_spam_distribution = ff.create_distplot([text_words_spam, text_words_ham],
                                               ['Spam','Ham'],
                                               curve_type='normal',
                                               colors=colors)
    fig_spam_distribution.update_layout(title_text="Distribution of word length in texts where target is 'spam' and 'ham'")
    fig_spam_distribution.update_layout(xaxis_title="Length of Message",
                                        yaxis_title="Probability of distribution",
                                        legend_title="Type of sms",
                                        font=dict(
                                            size=14,
                                            )
                                        )
    graphJSON_distribution = json.dumps(fig_spam_distribution, cls=plotly.utils.PlotlyJSONEncoder)
    del colors
    del text_words_ham
    del text_words_spam
    del fig_spam_distribution
    
    # for bar chart
    count_in_spam = get_dict_count(data=data, class_of_label=1)
    fig_count_in_spam = px.bar(count_in_spam,
                               y='Word',
                               x='Count',
                               color='Word',
                               text_auto='2s.',
                               title="Top 15 common words in label `Spam`")
    graphJSON_15_count_spam = json.dumps(fig_count_in_spam, cls=plotly.utils.PlotlyJSONEncoder)
    count_in_ham = get_dict_count(data=data, class_of_label=0)
    fig_count_in_ham = px.bar(count_in_ham,
                               y='Word',
                               x='Count',
                               color='Word',
                               text_auto='2s.',
                               title="Top 15 common words in label `Ham`")
    graphJSON_15_count_ham = json.dumps(fig_count_in_ham, cls=plotly.utils.PlotlyJSONEncoder)
    del count_in_spam
    del count_in_ham
    
    # bar plot for statistic model
    cur.execute("""SELECT CONCAT(mts.id_train, " ", mi.model_name, " ", model_class),
                mts.id_dgroup, can_use, time_train, accuracy_model_train
                FROM model_train_state mts
                JOIN model_train mt ON mts.id_train = mt.id_train
                JOIN model_info mi ON mi.id_model = mt.id_model
                WHERE id_dgroup = %s
                ORDER BY accuracy_model_train DESC
                LIMIT 5""", 
                (id_dgroup, ))
    records = cur.fetchall()
    columnName = ['info_name', 'id_dgroup', 'can_use', 'time_train', 'accuracy_model_train']
    data_models = pd.DataFrame.from_records(records, columns=columnName)
    fig_acc_train = px.bar(data_models,
                    y='info_name',
                    x='accuracy_model_train',
                    color='info_name',
                    text='accuracy_model_train',
                    title="Top 5 model with high accuracy train dataset")
    fig_acc_train.update_layout(xaxis_title="Accuracy score",
                                yaxis_title="Model name",
                                legend_title="Model name")
    graphJSON_acc_train = json.dumps(fig_acc_train, cls=plotly.utils.PlotlyJSONEncoder)
    cur.execute("""SELECT CONCAT(mts.id_train, " ", mi.model_name, " ", model_class),
                mts.id_dgroup, can_use, time_train, accuracy_model_test
                FROM model_train_state mts
                JOIN model_train mt ON mts.id_train = mt.id_train
                JOIN model_info mi ON mi.id_model = mt.id_model
                WHERE id_dgroup = %s
                ORDER BY accuracy_model_test DESC
                LIMIT 5""", 
                (id_dgroup, ))
    records = cur.fetchall()
    columnName = ['info_name', 'id_dgroup', 'can_use', 'time_train', 'accuracy_model_test']
    data_models = pd.DataFrame.from_records(records, columns=columnName)
    fig_acc_test = px.bar(data_models,
                    y='info_name',
                    x='accuracy_model_test',
                    color='info_name',
                    text='accuracy_model_test',
                    title="Top 5 model with high accuracy test dataset")
    fig_acc_test.update_layout(xaxis_title="Accuracy score",
                                yaxis_title="Model name",
                                legend_title="Model name")
    graphJSON_acc_test = json.dumps(fig_acc_test, cls=plotly.utils.PlotlyJSONEncoder)
    cur.execute("""SELECT CONCAT(mts.id_train, " ", mi.model_name, " ", model_class),
                mts.id_dgroup, can_use, time_train, accuracy_model_test
                FROM model_train_state mts
                JOIN model_train mt ON mts.id_train = mt.id_train
                JOIN model_info mi ON mi.id_model = mt.id_model
                WHERE id_dgroup = %s
                ORDER BY time_train ASC
                LIMIT 5""", 
                (id_dgroup, ))
    records = cur.fetchall()
    columnName = ['info_name', 'id_dgroup', 'can_use', 'time_train', 'accuracy_model_test']
    data_models = pd.DataFrame.from_records(records, columns=columnName)
    fig_time_train = px.bar(data_models,
                    y='info_name',
                    x='time_train',
                    color='info_name',
                    text='time_train',
                    title="Top 5 model with high fastest training time")
    fig_time_train.update_layout(xaxis_title="Time training (s)",
                                yaxis_title="Model name",
                                legend_title="Model name")
    graphJSON_time_train= json.dumps(fig_time_train, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template(session['role'] + "/view_one_group_info.html",
                           graphJSON_pie = graphJSON_pie,
                           graphJSON_distribution = graphJSON_distribution,
                           graphJSON_15_count_spam = graphJSON_15_count_spam,
                           graphJSON_15_count_ham = graphJSON_15_count_ham,
                           graphJSON_acc_train = graphJSON_acc_train,
                           graphJSON_acc_test = graphJSON_acc_test,
                           graphJSON_time_train = graphJSON_time_train,
                           num_rows = num_rows,
                           group_info = check_exist[0],
                           userinfo=session['username'])

@app.route("/view_model_info")
def view_model_info():
    cur = mysql.connection.cursor()
    sql = """
                    SELECT *
                    FROM model_info 
                """
    cur.execute(sql)
    model_info = cur.fetchall()

    return render_template(session['role'] + "/view_model_info.html", data=model_info, userinfo=session['username'])

@app.route("/view_model_info/form_add_model_info", methods=['GET','POST'])
def form_add_model_info():
    cur = mysql.connection.cursor()

    if request.method == 'POST':
        details = request.form
        model_name = details['model_name'].strip()
        model_class = details['model_class'].strip()
        description = details['description'].strip()

        cur.execute("INSERT INTO model_info (model_name, model_class, description) VALUES (%s, %s, %s)",
                    (model_name, model_class, description))
        mysql.connection.commit()
        cur.close()

        flash("Thêm mô hình thành công !!!")
        return redirect(url_for('view_model_info'))

    return render_template(session['role'] + "/form_add_model_info.html", userinfo=session['username'])


@app.route("/view_model_info/update_one_model_info/<int:id_model>", methods=['GET','POST'])
def update_one_model_info(id_model):
    cur = mysql.connection.cursor()
    sql = """
                SELECT * 
                from model_info
                WHERE model_info.id_model = %s;
              """
    cur.execute(sql, (id_model,))
    one_model_info = cur.fetchone()

    if request.method == 'POST':
        details = request.form
        model_name = details['model_name'].strip()
        model_class = details['model_class'].strip()
        description = details['description'].strip()

        cur.execute("UPDATE model_info SET model_name = %s, model_class = %s, description = %s WHERE id_model = %s",
                    (model_name, model_class, description, id_model))
        mysql.connection.commit()
        cur.close()

        flash("Chỉnh sửa thành công !!!")
        return redirect(url_for('view_model_info'))

    return render_template(session['role'] + "/update_one_model_info.html", data=one_model_info, userinfo=session['username'])

# chưa có thông báo xác nhận xóa không
@app.route("/delete_one_model_info/<int:id_model>")
def delete_one_model_info(id_model):
    cur = mysql.connection.cursor()
    sql = """
            SELECT * 
            from model_train
            WHERE model_train.id_model = %s;
          """
    cur.execute(sql, (id_model,))
    records = cur.fetchall()

    if len(records) != 0:
        flash("Không thể xóa model này!!")
    else:
        sql = """
                DELETE FROM model_info
                WHERE id_model = %s;
              """
        cur.execute(sql, (id_model,))
        mysql.connection.commit()
        flash("Đã xóa thành công model có ID: " + str(id_model))
    cur.close()
    return redirect(url_for("view_model_info"))

@app.route("/view_model_info/view_model_train_state")
def view_model_train_state():
    cur = mysql.connection.cursor()
    sql = """
                        SELECT *
                        FROM model_train_state 
                    """
    cur.execute(sql)
    model_train_state = cur.fetchall()

    return render_template(session['role'] + "/view_model_train_state.html", data=model_train_state, userinfo=session['username'])

##cần fix

@app.route("/view_model_info/view_model_train_state/update_one_model_train_state/<int:id_train>", methods=['GET','POST'])
def update_one_model_train_state(id_train):
    cur = mysql.connection.cursor()
    sql = """
                    SELECT * 
                    from model_train_state
                    WHERE model_train_state.id_train = %s;
                  """
    cur.execute(sql, (id_train,))
    one_model_train_state = cur.fetchall()
    
    if len(one_model_train_state) == 0:
        return "Error"

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
    
    data_group = [elm for elm in data_group]
    for i in range(len(data_group)):
        if data_group[i][0] == one_model_train_state[0][1]:
            temp = data_group[0]
            data_group[0] = data_group[i]
            data_group[i] = temp
            break

    if request.method == 'POST':
        details = request.form
        time_train = details['time_train_update']
        can_use = 1 if 'can_use_update' in details.keys() else 0

        # take id_dgroup
        id_data_choice = details['info_data_update']

        # accuracy model
        acc_model_train = details['acc_model_train_update']
        acc_model_test = details['acc_model_test_update']

        now = datetime.now()
        update_at = now.strftime("%Y-%m-%d %H:%M:%S")

        sql_update = """
                    UPDATE model_train_state SET id_dgroup = %s, can_use = %s, time_train = %s,
                     accuracy_model_train = %s, accuracy_model_test = %s, update_by = %s, update_at = %s
                     WHERE id_train = %s
                """

        cur.execute(sql_update, (id_data_choice, can_use, time_train, acc_model_train, acc_model_test,
                          session['username'][1], update_at, id_train))

        mysql.connection.commit()

        # model train table
        model_info = details['model_info_update']
        for i in range(len(model_info_choice)):
            if model_info == model_info_choice[i]:
                model_info = model_infos[i][0]

        cur.execute("""
                    UPDATE model_train SET id_model = %s
                    WHERE id_train = %s
                    """, (model_info, id_train))
        mysql.connection.commit()
        flash("Chỉnh sửa thành công !!!")
        return redirect(url_for('view_model_train_state'))

    return render_template(session['role'] + "/update_one_model_train_state.html",
                           current_data=one_model_train_state[0],
                           model_infos=model_info_choice,
                           data_group=data_group,
                           userinfo=session['username'])

# chưa có thông báo xác nhận xóa không
@app.route("/delete_one_model_train_state/<int:id_train>")
def delete_one_model_train_state(id_train):
    cur = mysql.connection.cursor()

    sql = """
        SELECT * FROM `model_train_state` WHERE model_train_state.id_train = %s
    """

    cur.execute(sql, (id_train, ))
    path = cur.fetchall()
    
    if len(path) == 0:
        return "Error"
    
    path = path[0]
    
    os.remove(path[2])

    sql_train_state = """
                DELETE FROM model_train_state
                WHERE id_train = %s;
              """
    sql_train = """
                    DELETE FROM model_train
                    WHERE id_train = %s;
                  """

    cur.execute(sql_train, (id_train,))
    cur.execute(sql_train_state, (id_train,))

    mysql.connection.commit()
    flash("Đã xóa thành công model train state có ID: " + str(id_train))
    cur.close()
    return redirect(url_for("view_model_train_state"))

@app.route("/view_one_model_train_state/<string:id_train>", methods=['GET','POST'])
def view_one_model_train_state(id_train):
    cur = mysql.connection.cursor()
    
    cur.execute("""
                SELECT * 
                FROM model_train_state
                WHERE id_train = %s
                """, (id_train, ))
    state_info = cur.fetchall()
    if len(state_info) == 0:
        return "Error"
    
    # model info
    cur.execute("""
                SELECT * 
                FROM model_info
                WHERE id_model = (
                    SELECT mt.id_model
                    FROM model_train mt
                    WHERE mt.id_train = %s
                    LIMIT 1
                )
                """, (id_train, ))
    model_info = cur.fetchall()[0]
    
    # select data to preprocessing
    cur.execute("""
                SELECT text, class_id
                FROM data_train
                WHERE id_dtrain IN (
                    SELECT dgs.id_dtrain
                    FROM data_group_split dgs
                    WHERE dgs.id_dgroup = %s
                )
                ORDER BY create_at ASC
                """, (state_info[0][1], ))
    records = cur.fetchall()
    columnName = ['Text', 'Target']
    data_input = pd.DataFrame.from_records(records, columns=columnName)
    
    # test size
    cur.execute("""
                SELECT dgi.test_size, dgi.tfidf_path, dgi.group_name
                FROM data_group_info dgi
                WHERE dgi.id_dgroup = %s
                """, (state_info[0][1], ))
    test_size, path_to_tfidf, group_data_name = cur.fetchall()[0]
    
    tfidf = pickle.load(open(path_to_tfidf, 'rb'))
    X = tfidf.transform(data_input['Text'].values)
    
    train_info, test_info = take_info_output(X, data_input['Target'].values,
                                             state_info[0][2], test_size)
    
    return render_template(session['role'] + "/view_one_model_train_state.html",
                           model_info = model_info,
                           train_info = train_info,
                           test_info = test_info,
                           state_info = state_info,
                           group_data_name = group_data_name,
                           test_size = test_size,
                           userinfo=session['username'])

# @app.route("/view_data_group/<string:group_id>_<string:mode>", methods=['GET','POST'])
# def view_data_group(group_id, mode):
#     return None

@app.route("/check_data_input", methods=['GET','POST'])
def check_data_input():
    return None

@app.route("/view_data_input")
def view_data_input():
    cur = mysql.connection.cursor()
    if session['role'] == "admin":
        sql = """
                        SELECT *
                        FROM data_input 
                    """
        cur.execute(sql)
        records = cur.fetchall()
    else:
        sql = """
                    SELECT *
                    FROM data_input
                    WHERE id_user = %s
                """
        cur.execute(sql, (session['username'][0], ))
        records = cur.fetchall()

    return render_template(session['role'] + "/view_data_input.html", data=records, userinfo=session['username'])

@app.route("/view_distinct_data_input")
def view_distinct_data_input():
    cur = mysql.connection.cursor()
    sql = """ 
            SELECT MIN(id) AS id, id_user, original_text, create_by
            FROM data_input
            GROUP BY id_user, original_text, create_by;
            """
    cur.execute(sql)
    records = cur.fetchall()
    return render_template(session['role'] + "/view_distinct_data_input.html", data=records, userinfo=session['username'])

@app.route("/delete_one_data_input/<int:id_data_input>", methods=['GET','POST'])
def delete_one_data_input(id_data_input):
    cur = mysql.connection.cursor()
    sql = """
            SELECT * FROM data_input WHERE data_input.id = %s;
        """
    cur.execute(sql, (id_data_input, ))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"

    sql_delete_input_data = """
                       DELETE FROM data_input
                       WHERE data_input.id = %s;
                     """
    cur.execute(sql_delete_input_data, (id_data_input, ))

    mysql.connection.commit()
    flash("Đã xóa thành công Input data có ID :" + str(id_data_input))
    cur.close()
    return redirect(url_for("view_data_input"))

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

    return render_template(session['role'] + "/view_account.html", data=accounts, userinfo=session['username'])

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

        cur.execute("SELECT username FROM user")
        existing_usernames = [row[0] for row in cur.fetchall()]
        if user_name in existing_usernames:
            error = "Tên người dùng đã tồn tại trong hệ thống!"
            return render_template(session['role'] + '/form_add_account.html', error=error, list_role=list_role)

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
        return render_template(session['role'] + '/form_add_account.html', message=message,  list_role=list_role, userinfo=session['username'])

    return render_template(session['role'] + '/form_add_account.html', list_role=list_role, userinfo=session['username'])

@app.route("/update_one_account/<int:id_user>", methods=['GET','POST'])
def update_one_account(id_user):
    cur = mysql.connection.cursor()
    sql = """
                    SELECT * FROM user
                    join role_user on user.id = role_user.id_user
                    join role on role.role_id = role_user.id_role
                    WHERE user.id = %s
              """
    cur.execute(sql, (id_user,))
    one_account = cur.fetchall()

    if len(one_account) == 0:
        return "Error"

    if request.method == 'POST':
        details = request.form
        full_name = details['full_name'].strip()
        role = details['role'].strip()

        cur.execute("SELECT * FROM role WHERE role_name = %s",
                    (role, ))
        role_id = cur.fetchone()[0]

        cur.execute("UPDATE user SET full_name = %s WHERE id = %s",
                    (full_name, id_user))

        cur.execute("UPDATE role_user SET id_role = %s WHERE id_user = %s",
                    (role_id, id_user))

        mysql.connection.commit()
        cur.close()

        flash("Chỉnh sửa thành công !!!")
        return redirect(url_for('view_account'))

    return render_template(session['role'] + "/update_one_account.html", data=one_account[0], userinfo=session['username'])


@app.route("/delete_one_account/<int:id_user>")
def delete_one_account(id_user):
    cur = mysql.connection.cursor()

    sql = """
        SELECT * FROM user WHERE user.id = %s
    """

    cur.execute(sql, (id_user,))
    records = cur.fetchall()

    if len(records) == 0:
        return "Error"

    record = records[0]

    sql_delete_user_role = """
                   DELETE FROM role_user
                   WHERE id_user = %s;
                 """
    sql_delete_user = """
                       DELETE FROM user
                       WHERE id = %s;
                     """

    cur.execute(sql_delete_user_role, (id_user,))
    cur.execute(sql_delete_user, (id_user,))

    mysql.connection.commit()
    flash("Đã xóa thành công tài khoản có ID: " + str(id_user))
    cur.close()
    return redirect(url_for("view_account"))


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
    return render_template(session['role'] + "/form_add_data_train.html", userinfo=session['username'])

@app.route("/form_add_data_input_to_data_train/<string:id_data_input>", methods=['GET','POST'])
def form_add_data_input_to_data_train(id_data_input):
    cur = mysql.connection.cursor()
    
    cur.execute("""
                SELECT original_text
                FROM data_input
                WHERE id = %s
                """, (id_data_input, ))
    data = cur.fetchall()
    if len(data) == 0:
        return "Error"
    data = data[0][0]
    
    cur.execute("""SELECT *
                    FROM data_group_info
                """)
    data_group_info = cur.fetchall()
    
    if request.method == 'POST':
        detail = request.form
        data_clean = step_corpus_for_one_text(detail['text_original'].strip())
        class_id = detail['class_id']
        
        cur.execute("SELECT * FROM data_train WHERE text = %s", (data_clean, ))
        check = cur.fetchall()
        if (len(check) != 0):
            return "Error"
        
        lst_group= list(detail.keys())
        lst_group.remove('text_original')
        
        # template sql 
        sql_data_train_template = "INSERT INTO data_train(text, class_id, create_by, update_by) VALUES (%s, %s, %s, %s)"
        sql_select_data_train_id = "SELECT id_dtrain FROM data_train WHERE text = %s"

        cur.execute(sql_select_data_train_id, (data_clean, ))
        data_id = cur.fetchall()
        if len(data_id) == 0:
            cur.execute(sql_data_train_template, (data_clean, class_id,
                                                    session['username'][1],
                                                    session['username'][1]))
            mysql.connection.commit()
            cur.execute(sql_select_data_train_id, (data_clean, ))
            data_id = cur.fetchall()
        data_id = data_id[0][0]
        
        for elm in data_group_info:
            if elm[1] in lst_group:
                cur.execute("""
                            INSERT INTO data_group_split(id_dtrain, id_dgroup)
                            VALUES (%s, %s)
                            """, (data_id, elm[0]))
                mysql.connection.commit()
        
        return redirect(url_for('home'))
    
    return render_template(session['role'] + "/form_add_data_input_to_data_train.html",
                           id_data_input = id_data_input,
                           data = data,
                           data_group_info = data_group_info,
                           userinfo=session['username'])

# ---- ADMIN ONLY ----

# User using model
@app.route("/is_spam_or_ham", methods=['GET','POST'])
@app.route("/is_spam_or_ham/<string:id_train>", methods=['GET','POST'])
def is_spam_or_ham(id_train = ''):
    cur = mysql.connection.cursor()
    
    cur.execute("""
                SELECT mts.id_train, CONCAT(mts.id_train, "-",mi.model_name)
                FROM model_train_state mts
                JOIN model_train mt ON mt.id_train = mts.id_train
                JOIN model_info mi ON mi.id_model = mt.id_model
                WHERE can_use = 1
                """)
    model_trains = cur.fetchall()
    if len(model_trains) == 0:
        return "Error"
    
    if id_train == '':
        id_train = model_trains[0][0]
    
    if request.method == 'POST':
        details = request.form
        model_train_id = details['model_train_id']
        text_input = details['text_input'].strip()
        
        if len(text_input) == 0:
            return redirect(url_for("is_spam_or_ham", id_train = id_train))
        
        cur.execute("""
                    INSERT INTO data_input(id_user, original_text, create_by, update_by)
                    VALUES (%s, %s, %s, %s)
                    """, (session['username'][0], text_input,
                          session['username'][1], session['username'][1]))
        mysql.connection.commit()
        
        if "/" in text_input:
            text_input = text_input.replace("/","")
        
        return redirect(url_for("is_spam_or_ham_result", id_train = model_train_id, text = text_input, userinfo=session['username']))
    
    model_trains_list = []
    for i in range(len(model_trains)):
        if str(model_trains[i][0]) == str(id_train):
            model_trains_list.insert(0, model_trains[i])
        else:
            model_trains_list.append(model_trains[i])
    
    return render_template(session['role'] + "/is_spam_or_ham.html",
                           model_trains = model_trains_list,
                           id_train = id_train, userinfo=session['username'])
     
    
    
@app.route("/is_spam_or_ham_result/<string:id_train>_<string:text>", methods=['GET','POST'])
def is_spam_or_ham_result(id_train, text):
    text = quote(text)
    cur = mysql.connection.cursor()
    
    cur.execute("""
                SELECT tfidf_path, path_to_state
                FROM model_train_state mts
                JOIN data_group_info dgi ON dgi.id_dgroup = mts.id_dgroup
                WHERE mts.id_train = %s
                """, (id_train, ))
    tfidf_path = cur.fetchall()
    
    if len(tfidf_path) == 0:
        return "Error"
    
    path_to_state = tfidf_path[0][1]
    tfidf_path = tfidf_path[0][0]
    
    step_index = [elm for elm in range(5)]
    
    step_title = [
        "Cleaning : Replacing all non-alphabetic, converting to lowecase",
        "Tokenize",
        "Remove stopwords",
        "Lemmatize",
        "Join tokenize to text"
    ]
    
    step_data = []
    step_data.append(step_1_clean(text))
    step_data.append(step_2_tokenize(step_data[-1]))
    step_data.append(step_3_remove_stopwords(step_data[-1]))
    step_data.append(step_4_lemmatizer(step_data[-1]))
    step_data.append(step_5_join_text(step_data[-1]))
    
    input_data = step_6_numerize(step_data[-1], tfidf_path)
    
    y_pred = predict_model(path_to_state, input_data)
    
    if y_pred == 1:
        y_pred = "Spam"
    else:
        y_pred = "Ham"
    
    return render_template(session['role'] + "/is_spam_or_ham_result.html",
                           label_pred = y_pred,
                           step_index = step_index,
                           step_data = step_data,
                           step_title = step_title,
                           id_train = id_train,
                           userinfo=session['username'])


