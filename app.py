import pandas as pd  # Import pandas library

# Load the Excel file
file_path = "1.xlsx"
data = pd.read_excel(data = pd.read_excel(r"C:\streamlit\1.xlsx"))

# Display the first few rows of data
print(data.head())  # Prints the first 5 rows of the data
