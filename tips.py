# HTAP-3: Environment setup and initial data load
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Load the dataset
tips = sns.load_dataset('tips')
print("Original Dataset Shape:", tips.shape)
print("\nFirst 5 rows of original dataset:")
print(tips.head())

# Create a copy of the dataset to preserve the original.
tips_clean = tips.copy()
print("\nA copy of the dataset has been created.")

# HTAP-4: Clean data and create tip_pct column
# Remove any duplicate rows from the copied dataset.
tips_clean = tips_clean.drop_duplicates()
print(f"\nAfter removing duplicates: {tips_clean.shape}")

# Drop the 'size' column from the copied dataset.
tips_clean = tips_clean.drop('size', axis=1)
print(f"After dropping 'size' column: {tips_clean.shape}")

# Create a new column 'tip_pct' for tip percentage (tip / total_bill * 100).
tips_clean['tip_pct'] = (tips_clean['tip'] / tips_clean['total_bill']) * 100
print("\n'tip_pct' column created.")
print("\nFirst 5 rows of cleaned dataset:")
print(tips_clean.head())

# HTAP-5: Create heatmap, histogram, stacked bar, and scatter plot
# Generate a 2x2 grid of plots.
plt.figure(figsize=(15, 10))
plt.suptitle('HTAP-5: Initial Set of Plots', fontsize=16)

# Plot 1: Heatmap of correlation matrix for numerical data.
plt.subplot(2, 2, 1)
numeric_data = tips_clean.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Heatmap of Correlation Matrix')

# Plot 2: Histogram showing the distribution of 'total_bill'.
plt.subplot(2, 2, 2)
plt.hist(tips_clean['total_bill'], bins=20, edgecolor='black', alpha=0.7)
plt.title('Histogram of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')

# Plot 3: Stacked bar chart of transactions (Day vs Time).
plt.subplot(2, 2, 3)
pivot_data = tips_clean.groupby(['day', 'time']).size().unstack(fill_value=0)
pivot_data.plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Stacked Bar of Transactions: Day vs Time')
plt.xticks(rotation=45)

# Plot 4: Scatter plot of 'total_bill' vs 'tip'.
plt.subplot(2, 2, 4)
plt.scatter(tips_clean['total_bill'], tips_clean['tip'], alpha=0.6)
plt.title('Scatter Plot: Total Bill vs Tip')
plt.xlabel('Total Bill')
plt.ylabel('Tip')

plt.tight_layout()
plt.show()

# HTAP-6: Use lmplot to analyze tip percentage relationships
print("\n--- HTAP-6: Linear Relationship Analysis ---")
# Create an lmplot to compare tip percentage between smokers and non-smokers.
print("\n1. Smoker vs Non-Smoker Tip Percentage:")
sns.lmplot(x='total_bill', y='tip_pct', data=tips_clean, hue='smoker', aspect=1.5)
plt.title('Tip Percentage vs Total Bill by Smoking Status')
plt.show()
# Answer: Look at the slope of the lines. The steeper slope indicates a higher tip percentage.

# Create an lmplot to compare tip percentage between lunch and dinner.
print("\n2. Lunch vs Dinner Tip Percentage:")
sns.lmplot(x='total_bill', y='tip_pct', data=tips_clean, hue='time', aspect=1.5)
plt.title('Tip Percentage vs Total Bill by Time of Day')
plt.show()
# Answer: Compare the slopes and intercepts.

# Create a grid of lmplots (2x2) to compare tip percentage by gender for both lunch and dinner.
print("\n3. Gender Analysis across Lunch and Dinner:")
g = sns.lmplot(x='total_bill', y='tip_pct', data=tips_clean, row='time', col='sex', aspect=1.2)
g.fig.suptitle('Tip Percentage vs Total Bill by Gender and Time', y=1.02)
plt.show()
# Answer: Visually examine the four plots. Is one gender consistently higher in both times?

# HTAP-7: Calculate average tips, bills, and groupby statistics
print("\n--- HTAP-7: Aggregate Analysis ---")
# Calculate the global average tip and average bill amount.
avg_tip = tips_clean['tip'].mean()
print(f"1. Average Tip Amount: ${avg_tip:.2f}")

avg_bill = tips_clean['total_bill'].mean()
print(f"2. Average Bill Amount: ${avg_bill:.2f}")

# Calculate the average tip grouped by gender.
avg_tip_gender = tips_clean.groupby('sex')['tip'].mean()
print(f"3. Average Tip by Gender:\n{avg_tip_gender}")

# Calculate the average tip on Sunday, grouped by gender.
avg_tip_sun_gender = tips_clean[tips_clean['day'] == 'Sun'].groupby('sex')['tip'].mean()
print(f"4. Average Tip on Sunday by Gender:\n{avg_tip_sun_gender}")

# (Using original data) Find which party size gives the highest average tip percentage.
tips['tip_pct'] = (tips['tip'] / tips['total_bill']) * 100
max_tip_pct_size = tips.groupby('size')['tip_pct'].mean().idxmax()
print(f"5. Party size that gives the highest average tip percentage: {max_tip_pct_size}")

# HTAP-8: Create box plots for bill amount by day and gender
# Generate box plots for total_bill by day and by gender, side by side
plt.figure(figsize=(12, 5))
plt.suptitle('HTAP-8: Box Plots for Total Bill', fontsize=16)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

# Box Plot 1: Bill by Day
plt.subplot(gs[0])
sns.boxplot(x='day', y='total_bill', data=tips_clean)
plt.title('Box Plot: Total Bill by Day')
plt.xticks(rotation=45)

# Box Plot 2: Bill by Gender
plt.subplot(gs[1])
sns.boxplot(x='sex', y='total_bill', data=tips_clean)
plt.title('Box Plot: Total Bill by Gender')

plt.tight_layout()
plt.show()

# HTAP-9: Recreate box plots using only Matplotlib
print("\n--- HTAP-9: Matplotlib Box Plots ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('HTAP-9: Box Plots using Matplotlib', fontsize=16)

# Matplotlib Box Plot 1: Bill by Day
days = tips_clean['day'].unique()
data_by_day = [tips_clean[tips_clean['day'] == d]['total_bill'].values for d in days]
ax1.boxplot(data_by_day, labels=days)
ax1.set_title('Matplotlib Box Plot: Total Bill by Day')
ax1.tick_params(axis='x', rotation=45)

# Matplotlib Box Plot 2: Bill by Gender
genders = tips_clean['sex'].unique()
data_by_gender = [tips_clean[tips_clean['sex'] == g]['total_bill'].values for g in genders]
ax2.boxplot(data_by_gender, labels=genders)
ax2.set_title('Matplotlib Box Plot: Total Bill by Gender')

plt.tight_layout()
plt.show()
