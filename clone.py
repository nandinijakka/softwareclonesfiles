from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation
from nltk.corpus import stopwords
import nltk
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

main = tkinter.Tk()
main.title("An Automatic Advisor for Refactoring Software Clones Based on Machine Learning") #designing main screen
main.geometry("1300x1200")

global dataset
global X,Y
filename = []
word_vector = []
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fmeasure = []

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def cleanPost(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = ' '.join(tokens)
    return tokens

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return cleanPost(string.strip().lower())

def handleFile(filePath):
    with open(filePath, "r") as f:
        lines=f.readlines()
        words = ''
        for line in lines:
            if '*' not in line:
                cleanedLine = clean_str(line)
                cleanedLine = cleanedLine.strip()
                cleanedLine = cleanedLine.lower()
                words+=cleanedLine+" "
    f.close()        
    return words.strip()

def uploadDataset():
    word_vector.clear()
    filename.clear()
    text.delete('1.0', END)
    filePath = filedialog.askdirectory(initialdir=".")
    text.insert(END,filePath+" loaded\n\n")
    for root, dirs, directory in os.walk(filePath):
        for j in range(len(directory)):
            text.insert(END,"Processing Code File : "+str(root)+"/"+str(directory[j])+"\n")
            text.update_idletasks()
            code = handleFile(root+"/"+directory[j])
            word_vector.append(code)
            filename.append(directory[j])
    text.insert(END,"\n\nTotal files found in dataset is : "+str(len(filename))+"\n") 
            

def featuresVector():
    global word_vector
    global filename
    text.delete('1.0', END)
    global X
    stopwords=stopwords = nltk.corpus.stopwords.words("english")
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords,use_idf=True, smooth_idf=False, norm=None, decode_error='replace')
    tfidf = tfidf_vectorizer.fit_transform(word_vector).toarray()        
    df = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    print(str(df))
    print(df.shape)
    df1 = df.values
    X = df1[:, 0:df1.shape[1]]
    X = np.asarray(X)
    filename = np.asarray(filename)
    word_vector = np.asarray(word_vector)
    text.insert(END,"Total Code Words Found in Repository : "+str(X.shape[1])+"\n")
    text.insert(END,"Generated Features\n\n");
    text.insert(END,str(df)+"\n")

def runLOF():
    global X_train, X_test, y_train, y_test
    global Y
    global X
    Y = []
    text.delete('1.0', END)
    clf = LocalOutlierFactor(n_neighbors=2, contamination=.1)
    outlier = clf.fit_predict(X)
    features = ''
    for i in range(len(outlier)):
        features+=str(outlier[i])+" "
    text.insert(END,features+"\n")

    for i in range(len(X)):
        count = 0
        for j in range(len(X[i])):
            if X[i,j] > 5:
                count = count + 1
        sim = count / len(word_vector[i].split(" "))
        if sim > 0.15:
            Y.append(1)
        else:
            Y.append(0)
    Y = np.asarray(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"\nMachine Learnig Training Records : "+str(len(X_train))+"\n")
    text.insert(END,"\nMachine Learnig Testing Records : "+str(len(X_test))+"\n")     

    

def runKNN():
    accuracy.clear()
    precision.clear()
    recall.clear()
    fmeasure.clear()
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test) 
    acc = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"KNN Accuracy : "+str(acc)+"\n")
    text.insert(END,"KNN Precision : "+str(p)+"\n")
    text.insert(END,"KNN Recall : "+str(r)+"\n")
    text.insert(END,"KNN FMeasure : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fmeasure.append(f)

def runRandomForest():
    global X_train, X_test, y_train, y_test
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test) 
    acc = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Random Forest Accuracy  : "+str(acc)+"\n")
    text.insert(END,"Random Forest Precision : "+str(p)+"\n")
    text.insert(END,"Random Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Random Forest FMeasure  : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fmeasure.append(f)

def runSVM():
    global X_train, X_test, y_train, y_test
    rf = svm.SVC()
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test) 
    acc = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"SVM Accuracy  : "+str(acc)+"\n")
    text.insert(END,"SVM Precision : "+str(p)+"\n")
    text.insert(END,"SVM Recall    : "+str(r)+"\n")
    text.insert(END,"SVM FMeasure  : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fmeasure.append(f)       


def runBagging():
    global X_train, X_test, y_train, y_test
    rf = BaggingClassifier(base_estimator=svm.SVC(), n_estimators=3, random_state=0)
    rf.fit(X_train, y_train)
    predict = rf.predict(X_test) 
    acc = accuracy_score(y_test,predict) * 100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    text.insert(END,"Bagging Classifier Accuracy  : "+str(acc)+"\n")
    text.insert(END,"Bagging Classifier Forest Precision : "+str(p)+"\n")
    text.insert(END,"Bagging Classifier Forest Recall    : "+str(r)+"\n")
    text.insert(END,"Bagging Classifier Forest FMeasure  : "+str(f)+"\n\n")
    accuracy.append(acc)
    precision.append(p)
    recall.append(r)
    fmeasure.append(f)

def graph():    

    df = pd.DataFrame([['KNN','Precision',precision[0]],['KNN','Recall',recall[0]],['KNN','F1 Score',fmeasure[0]],['KNN','Accuracy',accuracy[0]],
                       ['SVM','Precision',precision[1]],['SVM','Recall',recall[1]],['SVM','F1 Score',fmeasure[1]],['SVM','Accuracy',accuracy[1]],
                       ['Bagging Classifier','Precision',precision[2]],['Bagging Classifier','Recall',recall[2]],['Bagging Classifier','F1 Score',fmeasure[2]],['Bagging Classifier','Accuracy',accuracy[2]],
                       ['Random Forest','Precision',precision[3]],['Random Forest','Recall',recall[3]],['Random Forest','F1 Score',fmeasure[3]],['Random Forest','Accuracy',accuracy[3]],
                     
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()    
    
    
def refactorAdvisor():
    global Y
    global filename
    text.delete('1.0', END)
    for i in range(len(filename)):
        fname = filename[i]
        advisor = Y[i]
        if advisor == 1:
            text.insert(END,"Refactor Required for "+fname+"\n")
    


font = ('times', 16, 'bold')
title = Label(main, text='An Automatic Advisor for Refactoring Software Clones Based on Machine Learning')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Code Repository Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=900,y=100)
uploadButton.config(font=font1)  

featuresButton = Button(main, text="Generate Features Vector", command=featuresVector, bg='#ffb3fe')
featuresButton.place(x=900,y=150)
featuresButton.config(font=font1) 

lofButton = Button(main, text="Calculate Local Outlier Factor", command=runLOF, bg='#ffb3fe')
lofButton.place(x=900,y=200)
lofButton.config(font=font1) 

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN, bg='#ffb3fe')
knnButton.place(x=900,y=250)
knnButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM, bg='#ffb3fe')
svmButton.place(x=900,y=300)
svmButton.config(font=font1)

bagButton = Button(main, text="Run Bagging Algorithm", command=runBagging, bg='#ffb3fe')
bagButton.place(x=900,y=350)
bagButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRandomForest, bg='#ffb3fe')
rfButton.place(x=900,y=400)
rfButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=900,y=450)
graphButton.config(font=font1)

refactorButton = Button(main, text="Refactor Software Advisor", command=refactorAdvisor, bg='#ffb3fe')
refactorButton.place(x=900,y=500)
refactorButton.config(font=font1)

main.config(bg='RoyalBlue2')
main.mainloop()
