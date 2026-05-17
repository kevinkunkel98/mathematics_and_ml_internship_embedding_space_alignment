import os 
import pandas as pd  
import matplotlib.pyplot as plt 

Current_dir = os.path.dirname(os.path.realpath(__file__))
path_df = os.path.join(Current_dir, "..", "data", "cat", "cat.csv")
df = pd.read_csv(path_df)

# Count breeds
df_num = df['Breed'].value_counts()

# Threshold
threshold = 1000

# Separate large and small categories
large = df_num[df_num >= threshold]
small = df_num[df_num < threshold]

df_plot = large.copy()
df_plot['Other'] = small.sum()

# Prepare data for the pie chart
counts = df_plot.values
labels = df_plot.index

# Save path
path_fig = os.path.join(Current_dir, "..", "visualization", "breed_pie.png")

# Create plot
plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 14})
plt.title("Breed Distribution", fontsize=16)

# Save BEFORE showing
plt.savefig(path_fig, dpi=300, bbox_inches='tight')
plt.show()