from tkinter import *
import tkinter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import preprocessing
window = tkinter.Tk()#This method creates a blank window with close, maximize, and minimize buttons on the top as a usual GUI should have
window.title("GUI")
# label = tkinter.Label(window,text="Welcome").pack()
# top_frame = tkinter.Frame(window).pack()
df = pd.read_csv('.vscode/surveylungcancer.csv')
#min_sample_split so mau toi thieu de phan chia (Moi mot ,Node phai co toi thieu min_sample_split)
print(df.shape)
lb = preprocessing.LabelEncoder()
data = df.apply(lb.fit_transform)
dt_Train,dt_Test = train_test_split(data,test_size=0.3,shuffle=True)
X_train = dt_Train.iloc[:, :15]
y_train = dt_Train.iloc[:, 15]
X_test = dt_Test.iloc[:, :15]
y_test = dt_Test.iloc[:, 15]
y_test = np.array(y_test)
print(X_train)
pla = Perceptron(fit_intercept=False,alpha=0.05).fit(X_train.values,y_train)
myHeader = ["GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE","FATIGUE" ,"ALLERGY","WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH","SWALLOWING DIFFICULTY","CHEST PAIN"]
varList = [StringVar() for i in range(0,len(myHeader))]
for i in range(0,len(myHeader)):
    tkinter.Label(window,text=myHeader[i]).grid(row=i,column=0)
    tkinter.Entry(window,textvariable=varList[i]).grid(row=i,column=1)
def retrive_input():
    res = []
    for i in range(0,len(varList)):
        # print(varList[i].get())
        data = int(varList[i].get())
        res.append(data)
    y_predict = pla.predict(np.array(res).reshape(1,15))
    lbl =tkinter.Label(window, text = y_predict).grid(row=16,column=0)
    print("1")
    
    # lbl.config(text="Giá trị dự đoán: "+y_predict)
tkinter.Button(window,text="Click me",command=retrive_input).grid(row=15,column=0)
# txt1 = tkinter.Text(top_frame) 
window.mainloop()
