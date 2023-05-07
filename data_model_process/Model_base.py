from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn import metrics


import pickle

def predict_model(path_to_state, input):
    model = pickle.load(open(path_to_state, 'rb')) # "../" + 
    
    y_pred = model.predict(input)
    
    return y_pred[0]

def take_info_output(X, y, path_to_state, test_size):
    model = pickle.load(open(path_to_state, 'rb')) # "../" + 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=42)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # accuracy_score, precision_score, recall_score, 
    # f1_score, metrics.r2_score, confusion matrix
    train_info = [
        accuracy_score(y_train, y_pred_train),
        precision_score(y_train, y_pred_train),
        recall_score(y_train, y_pred_train),
        metrics.r2_score(y_train, y_pred_train),
        metrics.f1_score(y_train, y_pred_train),
        metrics.confusion_matrix(y_train, y_pred_train).flatten().tolist()
    ]
    test_info = [
        accuracy_score(y_test, y_pred_test),
        precision_score(y_test, y_pred_test),
        recall_score(y_test, y_pred_test),
        metrics.r2_score(y_test, y_pred_test),
        metrics.f1_score(y_test, y_pred_test),
        metrics.confusion_matrix(y_test, y_pred_test).flatten().tolist()
    ]
    return train_info, test_info