import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score
from imblearn.under_sampling import RandomUnderSampler
import pickle

import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":

    df = pd.read_csv("./data/churn.csv")


    df_categoricas = df.select_dtypes(include=['object','category'])
    df_numericas = df.select_dtypes(include='number')

    colums = df_categoricas.columns
    print(colums)
    for x in colums:
        print(df[x].unique())

    colums_num = df_numericas.columns
    print(colums_num)
    print(df_numericas.describe())

    column_equivalence = {}
    features = list(df.columns)
    for i, column in enumerate(list([str(d) for d in df.dtypes])):
        if column == "object":
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].mode())
            categorical_column = df[df.columns[i]].astype("category")
            current_column_equivalence = dict(enumerate(categorical_column.cat.categories))
            column_equivalence[i] = dict((v,k) for k,v in current_column_equivalence.items())
            df[df.columns[i]] = categorical_column.cat.codes
        else:
            df[df.columns[i]] = df[df.columns[i]].fillna(df[df.columns[i]].median())

   

    X = df.drop('Exited' , axis=1)
    y = df['Exited']

    undersample = RandomUnderSampler(random_state=42)

    X = df.drop('Exited',axis=1)
    y = df.Exited

    X_over , y_over = undersample.fit_resample(X,y)

    X_train, X_test, y_train , y_test = train_test_split(X_over, y_over, random_state=42, shuffle=True, test_size= .2)

   
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_X_train= scaler.transform(X_train)
    scaled_X_test= scaler.transform(X_test)

    clf_rf = RandomForestClassifier(random_state=0)
    clf_rf.fit(X_train, y_train)

    y_pred = clf_rf.predict(X_test)

    print(confusion_matrix(y_test, clf_rf.predict(X_test)))

    f1 = f1_score(y_test, y_pred)
   
    accuracy = accuracy_score(y_test, y_pred)

    print('F1-score:', f1)
    print('Accuracy:', accuracy)

    cm = confusion_matrix(y_test, y_pred)

    # Visualizar la matriz de confusión como un mapa de calor
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, vmin=0, fmt='.0f', cbar=False, linewidths=.5, square=True, cmap='coolwarm')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('RandomForest')
    plt.savefig('figura_modelo_actualizado.png')
    plt.show()

    
    pickle.dump(clf_rf, open("churn/models/model.pk", "wb"))
    pickle.dump(column_equivalence, open("churn/models/column_equivalence.pk", "wb"))
    pickle.dump(features, open("churn/models/features.pk", "wb"))
    