import pandas as pd
import requests
import os
# URL des GitHub-Datensatzes
dataset_url = 'https://github.com/RiccardoDAndrea/Streamlit-Regression-App/raw/main/supermarket_sales%20-%20Sheet1.csv'

# Herunterladen des Inhalts der CSV-Datei
response = requests.get(dataset_url)
content = response.content

# Speichern des Inhalts als temporäre Datei
temp_file = 'temp.csv'
with open(temp_file, 'wb') as f:
    f.write(content)

# Laden der CSV-Datei mit Pandas
dataset = pd.read_csv(temp_file)
uploaded_file = pd.DataFrame(dataset)
# Ausgabe des Datensatzes
print(uploaded_file)


