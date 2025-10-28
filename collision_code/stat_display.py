import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the US Accidents dataset (this may take some time due to size)
df = pd.read_csv("data/US_Accidents_March23.csv")
# Peek at the first few rows
print(df.head(5))
# Basic summary statistics
print(df.describe())
# Check for missing values in each column
print(df.isnull().sum())
# Distribution of Severity values
severity_counts = df["Severity"].value_counts()
print(severity_counts)
print(severity_counts / len(df) * 100)  # percentages
sns.countplot(x="Severity", data=df)
plt.title("Accident Severity Distribution")
plt.show()
df["Start_Time"] = pd.to_datetime(
    df["Start_Time"], format="ISO8601", errors="coerce"
)
# Extract hour of day
df["Hour"] = df["Start_Time"].dt.hour
# Count accidents by hour
hour_counts = df["Hour"].value_counts().sort_index()
# Plot
plt.figure(figsize=(8, 4))
hour_counts.plot(kind="bar", color="skyblue")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.show()
