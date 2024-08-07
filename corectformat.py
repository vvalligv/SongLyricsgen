import re
import pandas as pd
import ast

# Load the dataset
df = pd.read_csv('preprocessed.csv')  # Provide the correct path to your CSV file

# Function to fix formatting issues
def fix_formatting(text):
    if isinstance(text, str):
        # Remove leading and trailing commas, spaces, and brackets
        text = text.strip('[], ')
        # Replace remaining spaces with commas
        text = re.sub(r'\s+', ',', text)
        # Ensure the list is enclosed in brackets
        text = f'[{text}]'
        return text
    return text

# Apply the formatting correction
df['padded_sequences'] = df['padded_sequences'].apply(fix_formatting)

# Convert the corrected strings back to lists using ast.literal_eval
df['padded_sequences'] = df['padded_sequences'].apply(lambda x: ast.literal_eval(x))

# Save the corrected dataset
df.to_csv('corrected_data.csv', index=False)
