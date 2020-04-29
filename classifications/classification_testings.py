import pandas as pd
import csv
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
#models
from nltk.classify import SklearnClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#metrics
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def nai_bay():
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    
    print("NaiveBayes:\n")
    print(classification_report(y_test, y_pred_gnb))
    print(confusion_matrix(y_test, y_pred_gnb))

def log_reg():
    lr = LogisticRegression(random_state=0, class_weight='balanced', solver="saga")
    lr.fit(X_train, y_train)
    lr_predict = lr.predict(X_test)

    print("LogisticRegression:\n")
    print(classification_report(y_test, lr_predict))
    print(confusion_matrix(y_test, lr_predict))

def dec_tree():
    dt = DecisionTreeClassifier(class_weight='balanced')
    dt.fit(X_train, y_train)
    dt_predict = dt.predict(X_test)

    print("DecisionTreeClassifier:\n")
    print(classification_report(y_test, dt_predict))
    print(confusion_matrix(y_test, dt_predict))

def random_forest():
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced')
    rf.fit(X_train, y_train)
    rf_predict = rf.predict(X_test)

    print("RandomForestClassifier:\n")
    print(classification_report(y_test, rf_predict))
    print(confusion_matrix(y_test, rf_predict))

def extra_trees():
    et = ExtraTreesClassifier()
    et.fit(X_train, y_train)
    et_pred = et.predict(X_test)

    print("ExtraTreesClassifier:\n")
    print(classification_report(y_test, et_pred))
    print(confusion_matrix(y_test, et_pred))

def svc():
    s_vc = SVC(class_weight="balanced", gamma="scale")
    s_vc.fit(X_train, y_train)
    s_vc_predict = s_vc.predict(X_test)

    print("SVC:\n")
    print(classification_report(y_test, s_vc_predict))
    print(confusion_matrix(y_test, s_vc_predict))

def lin_svc():
    lin_svc =  LinearSVC()
    lin_svc.fit(X_train, y_train)
    lin_svc_predict = lin_svc.predict(X_test)

    print("LinearSVC:\n")
    print(classification_report(y_test, lin_svc_predict))
    print(confusion_matrix(y_test, lin_svc_predict))

def neur_net():
    mlpc = MLPClassifier(hidden_layer_sizes=300, max_iter=500)
    mlpc.fit(X_train,y_train)
    y_pred_mlpc = mlpc.predict(X_test)

    print("MLPClassifier:\n")
    print(classification_report(y_test, y_pred_mlpc))
    print(confusion_matrix(y_test, y_pred_mlpc))

def lda_lin():
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    lda_pred = lda.predict(X_test)

    print("LinearDiscriminantAnalysis:\n")
    print(classification_report(y_test, lda_pred))
    print(confusion_matrix(y_test, lda_pred))

def qda_qua():
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    qda_pred = qda.predict(X_test)

    print("QuadraticDiscriminantAnalysis:\n")
    print(classification_report(y_test, qda_pred))
    print(confusion_matrix(y_test, qda_pred))

def knn_neigh():
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)

    print("KNeighborsClassifier:\n")
    print(classification_report(y_test, knn_pred))
    print(confusion_matrix(y_test, knn_pred))
    
    

if __name__ == "__main__":

    df_total = pd.read_csv("third_df_rmsw.csv") 
    #df_total = df_total.drop("isNegated", 1)
    #df_total = df_total.drop("isObject", 1)

    df_pos_neg = df_total.drop( df_total[df_total['polarity'] == 3 ].index)
    df_neg = df_total.drop( df_total[df_total['polarity'] != 2 ].index)
    df_pos = df_total.drop( df_total[df_total['polarity'] != 4 ].index)
    df_vneg = df_total.drop( df_total[df_total['polarity'] != 1 ].index)
    df_vpos = df_total.drop( df_total[df_total['polarity'] != 5 ].index)

    df_neu = df_total.drop( df_total[df_total['polarity'] != 3 ].index)
    
    #divison of neutral dataset into smaller ones
    df_neu1, df_neu3 = np.split(df_neu, [int(.8 * len(df_neu))])
    df_neu1, df_neu2 = np.split(df_neu1, [int(.5 * len(df_neu1))])
    df_neu3, df_neu4 = np.split(df_neu3, [int(.5 * len(df_neu3))])


    df_total1 = df_neu1.append(df_pos, ignore_index = True)
    df_total2 = df_neu2.append(df_neg, ignore_index = True)
    df_total3 = df_neu3.append(df_vneg, ignore_index = True)
    df_total4 = df_neu4.append(df_vpos, ignore_index = True)

    q = df_total['polarity'].astype(int)
    #y = np.where((q==2) | (q == 1), 0, 1)
    #print(sum(y == 0))
    #print(sum(y == 1))
    y = np.where(q == 3, 0, 1) #convert polarities binary: 0-neutral/1-polar
    X = df_total.drop("polarity",1) 


    ts=0.25
    rs=27
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = ts, random_state = rs)
    
    nai_bay()
    #dec_tree()
    random_forest()
    #extra_trees()
    log_reg()
    svc()
    #lin_svc()
    #neur_net()
    #lda_lin()
    
    #qda_qua()
    knn_neigh()

