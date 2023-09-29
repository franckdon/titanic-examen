# BASSA KOFFI FRANCK DONALD MASTER 2IA

import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

if __name__ == '__main__':
    project = ProjectInitializer()
    project.initialize_project()
    
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

    # Charger les données
