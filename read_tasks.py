import pandas as pd
import sys

try:
    df = pd.read_excel('task_distribution.xlsx')
    print(df.to_string())
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)
