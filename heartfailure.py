import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#importing the data
data= pd.read_csv (r"C:\Users\kcham\Downloads\archive\heart_failure_clinical_records_dataset.csv")

#seperating the data for analysis
categorical_variables = data[["anaemia", "diabetes", "high_blood_pressure","sex", "smoking"]]
continous_variables = data [['age', 'creatinine_phosphokinase', "ejection_fraction", "platelets", "serum_creatinine","serum_sodium", "time"]]

print (data.groupby("DEATH_EVENT").count())

#visualizing that data
age= data[["age"]]
platelets = data [["platelets"]]

plt.figure(figsize=(13,7))
plt.scatter(platelets, age, c=data["DEATH_EVENT"], s=100, alpha =0.8)
plt.xlabel("Platelets", fontsize=20)
plt.ylabel("Age", fontsize=20)
plt.title("Visualizing the unbalanced data", fontsize=22)
plt.show()

plt.figure(figsize=(13,7))
sns.heatmap(data.corr(), vmin=-1, vmax=1, cmap= "YlGnBu", annot= True)
plt.title("Relationship between all variables and DEATH EVENT", fontsize = 22)
plt.show()

plt.figure(figsize=(13,10))
for i, cat in enumerate(categorical_variables):
    plt.subplot(2,3,i+1)
    sns.countplot(data=data, x=cat, hue = "DEATH_EVENT")
plt.show()    

plt.figure(figsize=(17,15))
for j, con in enumerate(continous_variables):
    plt.subplot(3,3,j+1)
    sns.histplot(data=data, x= con, hue = "DEATH_EVENT", multiple = "stack")
plt.show()

#Smoking data
smokers = data[data["smoking"]==1]
non_smokers = data[data["smoking"]==0]

non_survived_smokers = smokers[smokers["DEATH_EVENT"]==1]
survived_non_smokers = non_smokers[non_smokers["DEATH_EVENT"]==0]
non_survived_non_smokers = non_smokers[non_smokers["DEATH_EVENT"]==1]
survived_smokers = smokers[smokers["DEATH_EVENT"]==0]

smoking_data = [len(non_survived_smokers), len(survived_non_smokers), len(non_survived_non_smokers), len(survived_smokers)]
smoking_labels = ["non_survived_smokers", "survived_non_smokers", "non_survived_non_smokers", "survived_smokers"]

plt.figure(figsize=(10,10))
plt.pie(smoking_data, labels = smoking_labels, autopct='%.1f%%', startangle=90)
circle= plt.Circle((0,0), 0.7, color = "white")
p= plt.gcf()
p.gca().add_artist(circle)
plt.title ("Survival status on smoking", fontsize = 22)
plt.show()

#Sex Data
male = data[data["sex"]==1]
female = data[data["sex"]==0]

non_survived_male= male[male["DEATH_EVENT"]==1]
survived_male = male[male["DEATH_EVENT"]==0]
non_survived_female = female[female["DEATH_EVENT"]==1]
survived_female = female[female["DEATH_EVENT"]==0]

sex_data= [len(non_survived_male), len(survived_male), len(non_survived_female), len(survived_female)]
sex_labels= ["non_survived_male", "survived_male", "non_survived_female", "survived_female"]

plt.figure(figsize=(9,9))
plt.pie(sex_data, labels = sex_labels, autopct= '%.1f%%', startangle=90)
circle= plt.Circle((0,0), 0.7, color = "white")
p= plt.gcf()
p.gca().add_artist(circle)
plt.title('Survivial status on sex', fontsize= 22)
plt.show()

#Diabetes Data
diabetes = data[data["diabetes"]==1]
no_diabetes = data[data["diabetes"]==0]

non_survived_diabetes = diabetes[diabetes["DEATH_EVENT"]==1]
survived_diabetes=  diabetes[diabetes["DEATH_EVENT"]==0]
non_survived_no_diabetes = no_diabetes[no_diabetes["DEATH_EVENT"]==1]
survived_no_diabetes= no_diabetes[no_diabetes["DEATH_EVENT"]==0]

diabetes_data= [len(non_survived_diabetes), len(survived_diabetes), len(non_survived_no_diabetes),len(survived_no_diabetes)]
diabetes_labels= ["non_survived_with_diabetes", "survived_with_diabetes", "non_survived_no_diabetes", "survived_no_diabetes"]

plt.figure(figsize=(9,9))
plt.pie(diabetes_data, labels = diabetes_labels, autopct= '%.1f%%', startangle=90)
circle= plt.Circle((0,0), 0.7, color = "white")
p= plt.gcf()
p.gca().add_artist(circle)
plt.title('Survivial status on Diabetes', fontsize= 22)
plt.show()

#Anaemia Data
anaemia= data[data["anaemia"]==1]
no_anaemia = data[data["anaemia"]==0]

non_survived_anaemia = anaemia[anaemia["DEATH_EVENT"]==1]
survived_anaemia = anaemia[anaemia["DEATH_EVENT"]==0]
non_survived_no_anaemia = no_anaemia[no_anaemia["DEATH_EVENT"]==1]
survived_no_anaemia = no_anaemia[no_anaemia["DEATH_EVENT"]==0]

anaemia_data= [len(non_survived_anaemia), len(survived_anaemia), len(non_survived_no_anaemia), len(survived_no_anaemia)]
anaemia_labels = ["non_survived_anaemia", "survived_anaemia", "non_survived_no_anaemia", "survived_no_anemia"]

plt.figure(figsize=(9,9))
plt.pie(anaemia_data, labels = anaemia_labels, autopct= '%.1f%%', startangle=90)
circle= plt.Circle((0,0), 0.7, color = "white")
p= plt.gcf()
p.gca().add_artist(circle)
plt.title('Survivial status on Anaemia', fontsize= 22)
plt.show()

#High Blood Pressure Data

high_blood_pressure= data[data["high_blood_pressure"]==1]
no_high_blood_pressure= data[data["high_blood_pressure"]==0]

non_survived_high_blood_pressure =high_blood_pressure[high_blood_pressure["DEATH_EVENT"]==1] 
survived_high_blood_pressure =high_blood_pressure[high_blood_pressure["DEATH_EVENT"]==0]
non_survived_no_high_blood_pressure =no_high_blood_pressure[no_high_blood_pressure["DEATH_EVENT"]==1]
survived_no_high_blood_pressure =no_high_blood_pressure[no_high_blood_pressure["DEATH_EVENT"]==0]

HBP_data = [len(non_survived_high_blood_pressure), len(survived_high_blood_pressure),len(non_survived_no_high_blood_pressure), len(survived_no_high_blood_pressure)]
HBP_labels = ["non_survived_High_blood_pressure", "survived_high_blood_pressure", "non_survived_no_high_blood_pressure", "survived_no_high_blood_pressure"]   

plt.figure(figsize=(9,9))
plt.pie(HBP_data, labels = HBP_labels, autopct= '%.1f%%', startangle=90)
circle= plt.Circle((0,0), 0.7, color = "white")
p= plt.gcf()
p.gca().add_artist(circle)
plt.title('Survivial status on High Blood Pressure', fontsize= 22)
plt.show()

#Data Modeling & prediction using continuous data:
x= data[["age", "creatinine_phosphokinase", "ejection_fraction", "serum_creatinine", "serum_sodium", "time"]]
y= data["DEATH_EVENT"]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state =2)

#data scaling
scaler = StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

accuracy_list = []

#Logistic Regression
lr_model= LogisticRegression()
lr_model.fit(x_train_scaled, y_train)
lr_prediction=lr_model.predict(x_test_scaled)
lr_accuracy=(round(accuracy_score(lr_prediction, y_test),4)*100)
accuracy_list.append(lr_accuracy)

#support Vector Machine
svc_model =  SVC()
svc_model.fit(x_train_scaled, y_train)
svc_prediction = svc_model.predict(x_test_scaled)
svc_accuracy = (round(accuracy_score(svc_prediction, y_test), 4)*100)
accuracy_list.append(svc_accuracy)

#KNearestNeighbor
knn_list= []
for k in range (1,50):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train_scaled, y_train)
    knn_prediction= knn_model.predict(x_test_scaled)
    knn_accuracy =  (round(accuracy_score(knn_prediction, y_test),4)*100)
k = np.arange (1,50)

knn_model = KNeighborsClassifier(n_neighbors=6)
knn_model.fit(x_train_scaled, y_train)
knn_prediction = knn_model.predict(x_test_scaled)
knn_accuracy = (round(accuracy_score(knn_prediction, y_test), 4)*100)
accuracy_list.append(knn_accuracy)

#Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion= "entropy", max_depth=2)
dt_model.fit(x_train_scaled, y_train)
dt_prediction=dt_model.predict(x_test_scaled)
dt_accuracy = (round(accuracy_score(dt_prediction, y_test),4)*100)
accuracy_list.append(dt_accuracy)

#Naive Bayes
nb_model = GaussianNB()
nb_model.fit(x_train_scaled, y_train)
nb_prediction = nb_model.predict(x_test_scaled)
nb_accuracy = (round(accuracy_score(nb_prediction, y_test),4)*100)
accuracy_list.append(nb_accuracy)

#Random Forest classifier
rf_model= RandomForestClassifier()
rf_model.fit(x_train_scaled, y_train)
rf_prediction=(rf_model.predict(x_test_scaled))
rf_accuracy= (round(accuracy_score(rf_prediction, y_test),4)*100)
accuracy_list.append(rf_accuracy)

print(accuracy_list)

models = ["Logistic Regression", "SVC", "KNearestNeighbors", "Decision Tree", "Naive Bayes", "Random Forest"]
#plotting the results
plt.figure(figsize=(12,7))
ax=sns.barplot(x=models, y= accuracy_list)
plt.xlabel("Classifiers", fontsize =15)
plt.ylabel("Accuracy (%)", fontsize=15)
for p in ax.patches:
    width = p.get_width()
    height= p.get_height()
    x=p.get_x()
    y=p.get_y()
    ax.annotate(f"{height} %", (x+width/2, y+ height*1.01), ha="center")
plt.show()