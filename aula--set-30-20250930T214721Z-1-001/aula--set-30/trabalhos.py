
import os
import pandas as pd


BASEDIR='/home/marcelo/Desktop/trabalhos-alunos'
RESULT='all.csv'

MYCOLS=['dataset', 'classifier', 'metric', 
        'v1', 'v2', 'v3', 'v4', 
        'v5', 'v6', 'v7', 'v8', 
        'v9', 'v10', 'v11', 'v12', 
        'v13', 'v14', 'v15', 'v16', 
        'v17', 'v18', 'v19', 'v20', 
        'author'
    ]

mylist = os.listdir(BASEDIR)
result = []
for fname in mylist:
    # se arquivo nao for csv, pula
    if fname[-3:] != 'csv': continue

    print(fname)
    df = pd.read_csv(BASEDIR+'/'+fname)
    df['author'] = fname[ :-4 ] # '.csv' --> 4 caracteres ;)
    df.columns = MYCOLS
    result.append(df)

result = pd.concat(result, axis=0, ignore_index=True)
result.to_csv(RESULT)



# ------------------ abaixo, valores de CLASSIFICADOR desregulado
# [
#   'perceptron' 'svm' 'bayes' 'trees' 'knn' 'Perceptron' 'SVM' 
#   'NaiveBayes' 'KNN' 'DecisionTree' 'Naive Bayes' 'Decision Tree' 
#   ' perceptron' ' svm' ' bayes' ' trees' ' knn' 'GaussianNB' 'SVC' 
#   'LogisticRegression' 'RandomForest' 'KNeighbors'
# ]


# ------------------ abaixo, valores de METRICAS desregulado
# [
#   'f1-score' 'accuracy' 'f1' 'F1-Score' 'Acurácia' 'ACC' 'F1' 
#   'f1_score' 'acc' ' f1' ' acc' 'F1-Measure' 'Accuracy' 'Acc' 
#   'F1_Score' 'Acuracia'
# ]
