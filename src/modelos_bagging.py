
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

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
   
    data = data.drop(data.columns[0:3], axis=1)

    print(data.head())

    # Convertimos los datos en formato categorico, para más info: shorturl.at/y0269
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

    print(data.head())
    print(column_equivalence)

    X = data.copy()
    y = X.pop(data.columns[-1])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.35, random_state=42)

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

    plt.savefig('figura_bagging.png')
    
    plt.show()

    




