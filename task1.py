import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore

#C:/Users/ikram/AppData/Local/Programs/Python/Python313/python.exe "c:/Users/ikram/Desktop/ISBID 2/stage/task1.py"

data_train = pd.read_csv(r'C:\Users\ikram\Desktop\tasks\house\train.csv')  
data_test = pd.read_csv(r'C:\Users\ikram\Desktop\tasks\house\test.csv')

print(data_train.head())



features = ['GrLivArea', 'TotRmsAbvGrd', 'FullBath']  # Surface en pieds carrés, nombre de chambres et salles de bain

# On suppose que 'SalePrice' est la variable cible dans le fichier train.csv
X_train = data_train[features]  # Caractéristiques d'entraînement
y_train = data_train['SalePrice']  # Variable cible d'entraînement

# 3. Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)


# 5. Faire des prédictions sur le jeu de test
X_test = data_test[features]  # Caractéristiques du test
predictions = model.predict(X_test)  # Prédictions des prix des maisons

# 6. Préparer le fichier de soumission
submission = pd.DataFrame({
    'Id': data_test['Id'],  # Assurez-vous que le fichier test.csv contient la colonne 'Id'
    'SalePrice': predictions
})

# Sauvegarder les résultats dans un fichier de soumission
submission.to_csv('submission.csv', index=False)

print("Le fichier de soumission a été créé avec succès.")