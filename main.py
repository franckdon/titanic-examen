# BASSA KOFFI FRANCK DONALD MASTER 2IA

import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class TitanicDataProcessor:
    def __init__(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
    def preprocess_data(self):
        self._feature_engineering()
        self._encode_categorical_features()
    
    def train_model(self):
        X = self.train_data.drop('Survived', axis=1)
        y = self.train_data['Survived']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'Accuracy: {accuracy}')
        
        return clf
    
    def train_model(self):
        X = self.train_data.drop('Survived', axis=1)
        y = self.train_data['Survived']
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f'Accuracy: {accuracy}')
        
        return clf
    
    def _feature_engineering(self, data=None):
        if data is None:
            data = self.train_data
        data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        bins = [0, 18, 35, 60, np.inf]
        labels = ['Child', 'Young Adult', 'Adult', 'Senior']
        data['AgeGroup'] = pd.cut(data['Age'], bins, labels=labels)

        data['FamilySize'] = data['SibSp'] + data['Parch']
        data['IsAlone'] = 0
        data.loc[data['FamilySize'] == 0, 'IsAlone'] = 1
    
        return data
    
    def _encode_categorical_features(self, data=None):
        if data is None:
            data = self.train_data
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        data['Title'] = label_encoder.fit_transform(data['Title'])
        return data
    
    def display_data_statistics(self, data):
        print("Taille du DataFrame :")
        print(data.shape)

        print("\nStatistiques descriptives pour les caractéristiques numériques :")
        print(data.describe())

        print("\nNombre de valeurs uniques pour les caractéristiques catégorielles :")
        categorical_features = data.select_dtypes(include=['object'])
        for column in categorical_features:
            unique_values = data[column].nunique()
            print(f"{column}: {unique_values}")

        print("\nValeurs manquantes par colonne :")
        missing_values = data.isnull().sum()
        print(missing_values[missing_values > 0])
    
    
class ProjectInitializer:
    def __init__(self):
        self.project_root = os.getcwd()

    def create_directory(self, directory_name):
        os.makedirs(os.path.join(self.project_root, directory_name), exist_ok=True)

    def create_file(self, file_name, content=""):
        with open(os.path.join(self.project_root, file_name), 'w') as file:
            file.write(content)

    def create_structure(self):

        self.create_directory('docs')
        self.create_file('LICENSE')
        self.create_file('.gitignore')
        self.create_directory('models')
        self.create_directory('notebooks')
        self.create_file('notebooks/main.ipynb', '{"cells": [{"cell_type": "code", "source": "import pandas as pd", "outputs": []}]}')

        self.create_file('README.md', 'Project README content')

        self.create_directory('reports')
        self.create_file('requirements.txt', 'pandas==1.2.3\nnumpy==1.21.0')

        self.create_directory('src')
        self.create_file('src/utils.py', 'def my_function():\n    pass')

    def initialize_project(self):
        self.create_structure()

    def commit_changes(self, message):
        subprocess.run(['git', 'add', '.'], cwd=self.project_root)
        subprocess.run(['git', 'commit', '-m', message], cwd=self.project_root)
    
    
    
def main():
    processor = TitanicDataProcessor('data/train_file.csv', 'data/test_file.csv')
    processor.preprocess_data()

if __name__ == '__main__':
    project = ProjectInitializer()
    project.initialize_project()
    main()
    
    
    tickets = [
        "Creation de la structure",
        "Creation des dossiers et fichiers",
        "Creation du dossier notebooks et main.ipynb",
        "Creation du README.md",
        "Creation  du dossier src et utils.py"
    ]

    for ticket in tickets:
        project.commit_changes(ticket)

    # Additional commits
    for i in range(5):
        project.commit_changes(f"Additional commit {i+1}")

    print("projet et commits terminé.")
    subprocess.call(['git', 'add', '.'])
    subprocess.call(['git', 'commit', '-m', 'Ajout de tous les fichiers'])
    subprocess.call(['git', 'push'])
