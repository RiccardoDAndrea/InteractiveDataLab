import pandas as pd
import requests
import os
# URL des GitHub-Datensatzes
dataset_url = 'https://github.com/RiccardoDAndrea/Streamlit-Regression-App/raw/main/supermarket_sales%20-%20Sheet1.csv'

# Herunterladen des Inhalts der CSV-Datei
def dataframe_from_url():
    # URL des GitHub-Datensatzes
    # Herunterladen des Inhalts der CSV-Datei

    dataset_raw_dict = { "Supermarket_dataset" : "https://raw.githubusercontent.com/RiccardoDAndrea/Streamlit-Regression-App/feature_dataset/Dataset/supermarket_sales%20.csv",
                        "Wage_dataset" : "https://raw.githubusercontent.com/RiccardoDAndrea/Streamlit-Regression-App/feature_dataset/Dataset/wage.csv"}
    if "Supermarket_dataset" in dataset_raw_dict:
        dataset = dataset_raw_dict["Supermarket_dataset"]
    elif "Wage_dataset" in dataset_raw_dict["Wage_dataset"]:
        dataset = dataset_raw_dict["Wage_dataset"]
    
    response = requests.get(dataset)
    content = response.content

    # Speichern des Inhalts als temporäre Datei
    temp_file = 'temp.csv'
    with open(temp_file, 'wb') as f:
        f.write(content)

    # Laden der CSV-Datei mit Pandas
    dataset = pd.read_csv(temp_file, sep = ";")
    os.remove(temp_file)
    dataset_regression = pd.DataFrame(dataset)
    return dataset_regression

dataset_under = dataframe_from_url()
dataset_under

df = pd.read_csv('/Users/riccardo/Library/CloudStorage/OneDrive-HSOS/Python/Datensätze/all_data.txt')
print(df.shape)





