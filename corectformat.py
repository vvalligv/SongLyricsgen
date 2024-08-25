import re #This module provides support for regular expressions, which are used for searching and manipulating strings.
import pandas as pd
import ast #Sometimes, data stored in files or obtained from user input might be in the form of a string that looks like a Python literal (e.g., a list or a dictionary).
#However, this string is not yet an actual Python object—it’s just a string representation of it. 
#ast.literal_eval() can safely parse this string and convert it into the actual Python object.

# Load the dataset
df = pd.read_csv('preprocessed.csv')  # Provide the correct path to your CSV file

# Function to fix formatting issues
def fix_formatting(text):
    if isinstance(text, str):
        # Remove leading and trailing commas, spaces, and brackets
        text = text.strip('[], ') #Removes any leading or trailing brackets, commas, or spaces from the string. For example, ", [1, 2, 3], " becomes "1, 2, 3".
        # Replace remaining spaces with commas
        text = re.sub(r'\s+', ',', text) #Replaces all spaces in the string with commas. This helps in converting something like "1 2 3" to "1,2,3".
        # Ensure the list is enclosed in brackets
        text = f'[{text}]' #Ensures the string is enclosed in brackets to represent it as a list. So, "1,2,3" becomes "[1,2,3]".
        return text
    return text
#Example:

#Input: ", [1 2 3], "
#Processing:
#Strip: "1 2 3"
#Replace spaces with commas: "1,2,3"
#Enclose in brackets: "[1,2,3]"
#Output: "[1,2,3]"

# Apply the formatting correction
df['padded_sequences'] = df['padded_sequences'].apply(fix_formatting)

# Convert the corrected strings back to lists using ast.literal_eval
df['padded_sequences'] = df['padded_sequences'].apply(lambda x: ast.literal_eval(x))
#ast.literal_eval(x): Safely evaluates the string and converts it back into an actual Python list.
#Example:
#Input: "[1,2,3]"
#Output: [1, 2, 3] (actual list)

# Save the corrected dataset
df.to_csv('corrected_data.csv', index=False)
#to_csv(''): Saves the corrected DataFrame back to a CSV file. You need to specify the file path where you want to save the cleaned data.

#index=False: Ensures that the DataFrame's index is not saved as a separate column in the CSV file.
