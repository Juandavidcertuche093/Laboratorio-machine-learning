import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score
from imblearn.under_sampling import RandomUnderSampler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":

    data = pd.read_csv('./data/churn.csv')

    df_categoricas = data.select_dtypes(include=['object','category'])
    df_numericas = data.select_dtypes(include='number')

    colums = df_categoricas.columns
    print(colums)
    for x in colums:
        print(data[x].unique())

    print('='*65)

    colums_num = df_numericas.columns
    print(colums_num)
    print(df_numericas.describe())

    column_equivalence = {}
    features = list(data.columns)
    for i, column in enumerate(list([str(d) for d in data.dtypes])):
        if column == "object":
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].mode())
            categorical_column = data[data.columns[i]].astype("category")
            current_column_equivalence = dict(enumerate(categorical_column.cat.categories))
            column_equivalence[i] = dict((v,k) for k,v in current_column_equivalence.items())
            data[data.columns[i]] = categorical_column.cat.codes
        else:
            data[data.columns[i]] = data[data.columns[i]].fillna(data[data.columns[i]].median())

    X = data.drop('Exited' , axis=1)
    y = data['Exited']

    undersample = RandomUnderSampler(random_state=42)

    X = data.drop('Exited',axis=1)
    y = data.Exited

    X_over , y_over = undersample.fit_resample(X,y)

    X_train, X_test, y_train , y_test = train_test_split(X_over, y_over, random_state=42, shuffle=True, test_size= .2)

    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_X_train= scaler.transform(X_train)
    scaled_X_test= scaler.transform(X_test)
   
    estimadores = {
        'logistic_regression ': LogisticRegression(),
        'desicion_tres': DecisionTreeClassifier(),
        'KNeighbors': KNeighborsClassifier(),
        'randon_forest':RandomForestClassifier(),
        'svc': SVC(),
        'LinearSCV': LinearSVC(),
        'SGDC': SGDClassifier(),
        'gradient_boost' : GradientBoostingClassifier(),
    }

    accuracies = []

    for name, estimador in estimadores.items():
        bagging = BaggingClassifier(base_estimator= estimador, n_estimators=7).fit(X_train, y_train)
        predictions = bagging.predict(X_test)
        accuracy = accuracy_score(predictions, y_test)
        accuracies.append(accuracy)

        print(f'accuracy Bagging whith {name}: {accuracy_score(predictions, y_test)}')



    plt.figure(figsize=(10, 6))
    plt.bar(estimadores.keys(), accuracies)
    plt.xlabel('Estimador')
    plt.ylabel('Precisión')
    plt.title('Precisión de los estimadores en Bagging')
    plt.xticks(rotation=20)
    plt.savefig('figura_modelos_bagging_2.png')
    plt.show()
   
    