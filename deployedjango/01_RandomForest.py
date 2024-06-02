import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,recall_score,roc_auc_score


diabet=pd.read_csv("data/diabetes.csv")

# Séparer les caractéristiques (features) de la variable cible (target)
X = diabet.drop('Outcome', axis=1)
y = diabet['Outcome'] #variable cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_features = int(np.sqrt(X.shape[1]))

# Créer le modèle de Random Forest
model = RandomForestClassifier(n_estimators=100, max_features=max_features, random_state=42)
# Entraîner le modèle
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accurcy = accuracy_score(y_train,y_train_pred)
test_accurcy = accuracy_score(y_test,y_test_pred)
train_auc = roc_auc_score(y_train,y_train_pred)
test_auc = roc_auc_score(y_test,y_test_pred)
train_recall = recall_score(y_train,y_train_pred)
test_recall = recall_score(y_test,y_test_pred)

performance_table = pd.DataFrame({
    'Metrique': ['Accurcy', 'AUC', 'Recall'],
    'Ensemble d\'entrainement': [train_accurcy, train_auc, train_recall],
    'Ensemble de test': [test_accurcy, test_auc, test_recall]
})

print(performance_table)

#Enregistrement du model
joblib.dump(model, 'model.pkl')
print("Model.pk")