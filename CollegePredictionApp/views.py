from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import pymysql
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


global uname, dataset, sc, rf_cls
accuracy = []
precision = []
recall = []
fscore = []
encoder = []
columns = ['gender', 'caste', 'region','branch', 'college']

def LoadDataset(request):
    if request.method == 'GET':
        global dataset
        dataset = pd.read_csv("CollegeDataset/Dataset.csv", nrows=2000)
        dataset.fillna(0, inplace = True)
        cols = dataset.columns
        output = '<table border="1" align="center"><tr>'
        font = '<font size="" color="black">'
        for i in range(len(cols)):
            output += '<th>'+font+cols[i]+'</th>'
        output += "</tr>"    
        dataset = dataset.values
        for i in range(dataset.shape[0]):
            output += "<tr>"
            for j in range(dataset.shape[1]):
                output += "<td>"+font+str(dataset[i,j])+"</td>"
        output += "</tr>"
        dataset = pd.read_csv("CollegeDataset/Dataset.csv", usecols=['rank', 'gender', 'caste', 'region', 'branch', 'college'], nrows=2000)
        dataset.fillna(0, inplace = True)
        context = {'data':output}
        return render(request, 'AdminScreen.html', context)

def calculateMetrics(algorithm, predict, y_test):
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def TrainML(request):
    if request.method == 'GET':
        global dataset, encoder,accuracy, precision, recall, fscore, sc, columns, rf_cls
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        encoder.clear()
        sc = MinMaxScaler(feature_range = (0, 1))
        for i in range(len(columns)):
            le = LabelEncoder()
            dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
            encoder.append(le)
        dataset1 = dataset.values
        X = dataset1[:,0:dataset1.shape[1]-1]
        Y = dataset1[:,dataset1.shape[1]-1]
        X = sc.fit_transform(X)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        X_train, X_test1, y_train, y_test1 = train_test_split(X, Y, test_size=0.1)

        rf_cls = RandomForestClassifier()
        rf_cls.fit(X_train, y_train)
        predict = rf_cls.predict(X_test)
        calculateMetrics("Random Forest", predict, y_test)

        svm_cls = SVC()
        svm_cls.fit(X_train, y_train)
        predict = svm_cls.predict(X_test)
        calculateMetrics("SVM", predict, y_test)

        dt_cls = DecisionTreeClassifier()
        dt_cls.fit(X_train, y_train)
        predict = dt_cls.predict(X_test)
        calculateMetrics("Decision Tree", predict, y_test)

        cols = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'FSCORE']
        output = '<table border="1" align="center"><tr>'
        font = '<font size="" color="black">'
        for i in range(len(cols)):
            output += '<th>'+font+cols[i]+'</th>'            
        output += "</tr>"
        algorithm = ['Random Forest', 'SVM', 'Decision Tree']
        for i in range(len(accuracy)):
            output += "<tr><td>"+font+str(algorithm[i])+"</td><td>"+font+str(accuracy[i])+"</td><td>"+font+str(precision[i])+"</td><td>"+font+str(recall[i])+"</td>"
            output += "<td>"+font+str(fscore[i])+"</td></tr>"
        context = {'data':output}
        return render(request, 'AdminScreen.html', context)

def PredictCollege(request):
    if request.method == 'GET':
        return render(request, 'PredictCollege.html', {})

def PredictCollegeAction(request):
    if request.method == 'POST':
        global dataset, sc, rf_cls, encoder
        rank = request.POST.get('t1', False)
        gender = request.POST.get('t2', False)
        caste = request.POST.get('t3', False)
        region = request.POST.get('t4', False)
        branch = request.POST.get('t5', False)

        testData = [int(rank), gender, caste, region, branch]
        temp = []
        temp.append(testData)
        temp = np.asarray(temp)
        print(temp.shape)
        df = pd.DataFrame(temp, columns=['rank', 'gender', 'caste', 'region', 'branch'])
        for i in range(len(encoder)-1):
            df[columns[i]] = pd.Series(encoder[i].transform(df[columns[i]].astype(str)))
    
        df = df.values
        df = sc.transform(df)
        predict = rf_cls.predict(df)
        print(predict)
        predict = encoder[4].inverse_transform(predict)
        context = {'data':"Predicted College for Admission : "+predict}
        return render(request, 'UserScreen.html', context)

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def AdminLogin(request):
    if request.method == 'GET':
       return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})
    

def Signup(request):
    if request.method == 'GET':
       return render(request, 'Signup.html', {})

def AdminLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            uname = username
            context= {'data':'welcome '+username}
            return render(request, 'AdminScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'ExpertLogin.html', context)
        
def UserLoginAction(request):
    global uname
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        index = 0
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '9248908912', database = 'CollegePrediction',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username,password FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username and password == row[1]:
                    uname = username
                    index = 1
                    break		
        if index == 1:
            context= {'data':'welcome '+uname}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)

def SignupAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        gender = request.POST.get('t4', False)
        email = request.POST.get('t5', False)
        address = request.POST.get('t6', False)
        output = "none"
        con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '9248908912', database = 'CollegePrediction',charset='utf8')
        with con:
            cur = con.cursor()
            cur.execute("select username FROM signup")
            rows = cur.fetchall()
            for row in rows:
                if row[0] == username:
                    output = username+" Username already exists"
                    break
        if output == 'none':
            db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '9248908912', database = 'CollegePrediction',charset='utf8')
            db_cursor = db_connection.cursor()
            student_sql_query = "INSERT INTO signup(username,password,contact_no,gender,email,address) VALUES('"+username+"','"+password+"','"+contact+"','"+gender+"','"+email+"','"+address+"')"
            db_cursor.execute(student_sql_query)
            db_connection.commit()
            print(db_cursor.rowcount, "Record Inserted")
            if db_cursor.rowcount == 1:
                output = 'Signup Process Completed'
        context= {'data':output}
        return render(request, 'Signup.html', context)
      
