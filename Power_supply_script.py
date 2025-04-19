#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


file_path = "C:/Users/enhan/Downloads/Expanded_Historical_Transformer_Failures (1).csv"





# In[26]:


df = pd.read_csv(file_path)
# Display the first few rows
print(df.head())


# In[ ]:





# In[27]:


# Check for missing values
print(df.isnull().sum())


# In[28]:


file_path_health = "C:/Users/enhan/Downloads/archive (1)/Health index1.csv"
df_healthID = pd.read_csv(file_path_health)





# In[29]:


# Check for missing values
print(df_healthID.isnull().sum())


# In[30]:


# Display the first few rows
print(df_healthID.head())


#                                                                         EDA

# In[31]:


# Print column names for both datasets
print("Columns in df_healthID (Health Index Dataset):")
print(df_healthID.columns)

print("\nColumns in df (Transformer Failures Dataset):")
print(df.columns)


# In[32]:


# Ensure df_healthID has Transformer IDs
df_healthID["Transformer ID"] = [f"TX-{i}" for i in range(1, len(df_healthID) + 1)]

# Display updated df_healthID
print(df_healthID.head())


# In[33]:


print("\nColumns in df (Transformer Failures Dataset):")
print(df.columns)

print("\nColumns in df_healthID (Health Index Dataset):")
print(df_healthID.columns)


# In[34]:


# Rename columns in df_healthID to match df
df_healthID.rename(columns={
    "Oxigen": "Oxygen",  # Fix spelling
    "Acethylene": "Acetylene (ppm)",  # Match df naming
    "Power factor": "Power Factor",
    "Interfacial V": "Interfacial Voltage (kV)",
    "Dielectric rigidity": "Dielectric Rigidity (kV)",
    "Water content": "Water Content (ppm)",
    "Health index": "Health Index"  # Match df naming
}, inplace=True)

# Display updated column names
print("\nUpdated Columns in df_healthID:")
print(df_healthID.columns)


# In[35]:


# Merge datasets on Transformer ID
df_merged = pd.merge(df, df_healthID, on="Transformer ID", how="inner")

# Display first few rows of the merged dataset
print("\nMerged Dataset:")
print(df_merged.head())


# In[36]:


print("\nColumns in df_merged:")
print(df_merged.columns)



# In[37]:


# Drop duplicate sensor columns and keep only one Health Index column
df_merged = df_merged.drop(columns=["Health Index_x", "Acetylene (ppm)_x", "Power Factor_x",
                                    "Interfacial Voltage (kV)_x", "Dielectric Rigidity (kV)_x",
                                    "Water Content (ppm)_x"])

# Rename the correct Health Index column
df_merged.rename(columns={"Health Index_y": "Health Index",
                          "Acetylene (ppm)_y": "Acetylene (ppm)",
                          "Power Factor_y": "Power Factor",
                          "Interfacial Voltage (kV)_y": "Interfacial Voltage (kV)",
                          "Dielectric Rigidity (kV)_y": "Dielectric Rigidity (kV)",
                          "Water Content (ppm)_y": "Water Content (ppm)"}, inplace=True)

# Display cleaned column names
print("\nUpdated Columns in df_merged:")
print(df_merged.columns)


# In[38]:


def classify_health_index(hi):
    if hi > 90:
        return "Healthy"
    elif 80 < hi <= 90:
        return "Warning"
    else:
        return "Critical"

# Apply classification to df_merged
df_merged["Health Risk Level"] = df_merged["Health Index"].apply(classify_health_index)

# Display first few rows of updated dataset
print("\nUpdated Dataset with Health Risk Levels:")
print(df_merged.head())


# In[39]:


# Check data types of df_merged columns
df_merged.dtypes


# In[40]:


# Convert Health Risk Level to categorical
df_merged["Health Risk Level"] = df_merged["Health Risk Level"].astype("category")

# Verify changes
print(df_merged.dtypes["Health Risk Level"])


# In[41]:


import matplotlib.pyplot as plt
import seaborn as sns

# Exclude non-numeric columns
numeric_df = df_merged.select_dtypes(include=["number"])

# Generate Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Sensor Readings")
plt.show()


# In[42]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 

# Display summary statistics
print("\nSummary Statistics:")
print(df_merged.describe())

# Plot distribution of Health Index
plt.figure(figsize=(8, 5))
sns.histplot(df_merged["Health Index"], bins=20, kde=True, color="blue")
plt.title("Distribution of Health Index")
plt.xlabel("Health Index")
plt.ylabel("Frequency")
plt.show()

# Box plot for detecting outliers in key sensor readings
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_merged[["Hydrogen", "Methane", "CO", "CO2", "Ethylene", "Ethane", "Acetylene (ppm)"]])
plt.title("Boxplot of Sensor Readings (Outlier Detection)")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_merged.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Sensor Readings")
plt.show()


# In[ ]:


get_ipython().system('pip install imbalanced-learn')


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Check class distribution of Health Risk Level
plt.figure(figsize=(8, 5))
sns.countplot(x=df_merged["Health Risk Level"], palette="viridis")
plt.title("Distribution of Health Risk Levels")
plt.xlabel("Health Risk Level")
plt.ylabel("Count")
plt.show()

# Display value counts
df_merged["Health Risk Level"].value_counts(normalize=True)


# In[47]:


import imblearn
print("imbalanced-learn version:", imblearn.__version__)


# In[46]:


from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split

# Extract features and target variable
features = ["Hydrogen", "Oxygen", "Nitrogen", "Methane", "CO", "CO2", "Ethylene",
            "Ethane", "Acetylene (ppm)", "DBDS", "Power Factor",
            "Interfacial Voltage (kV)", "Dielectric Rigidity (kV)", "Water Content (ppm)"]

X = df_merged[features]
y = df_merged["Health Risk Level"]

# Encode target variable (convert categories to numbers)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

# Convert back to dataframe for verification
df_resampled = pd.DataFrame(X_resampled, columns=features)
df_resampled["Health Risk Level"] = label_encoder.inverse_transform(y_resampled)

# Check class distribution after balancing
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.countplot(x=df_resampled["Health Risk Level"], palette="viridis")
plt.title("Balanced Distribution of Health Risk Levels (After SMOTE)")
plt.xlabel("Health Risk Level")
plt.ylabel("Count")
plt.show()

# Display new class distribution
print(df_resampled["Health Risk Level"].value_counts(normalize=True))



# Class Balance Analysis: Health Risk Level Distribution
# The distribution of Health Risk Levels is highly imbalanced, with the majority of transformers classified as "Critical", while "Healthy" and "Warning" cases are severely underrepresented.
# 
# Key Findings:
# Critical Condition Dominates → The dataset is mostly composed of transformers in critical condition, which could lead to a model that is biased toward predicting failures.
# Severe Class Imbalance → There are very few Healthy and Warning examples, which may result in poor predictive accuracy for these categories.
# Potential Model Issues:
# The model might overfit to predicting "Critical" since it sees this class the most.
# The Healthy and Warning classes might be underpredicted or ignored by the model.

# Exploratory Data Analysis (EDA) Results
# 1. Distribution of Health Index
# The Health Index distribution is skewed toward lower values, indicating that a significant portion of transformers have poor health.
# Most transformers fall in the 50-60 range, suggesting that the dataset primarily consists of transformers in warning or critical condition.
# Very few transformers have a Health Index above 80, which means there are limited examples of healthy transformers.
# 2. Boxplot Analysis – Detecting Outliers in Sensor Readings
# The boxplot of Hydrogen, Methane, CO, CO2, Ethylene, Ethane, and Acetylene (ppm) reveals a significant number of outliers in various gas concentrations.
# Hydrogen and CO2 levels show extreme values (some exceeding 25,000 ppm), which may indicate critical transformer failures or sensor anomalies.
# Other gases, such as Methane and Ethylene, have some outliers but remain within a more reasonable range.
# The presence of many outliers suggests that either:
# These are genuine failure cases (severe transformer degradation).
# There are possible sensor measurement errors or inconsistencies.
# 3. Correlation Heatmap – Relationship Between Sensor Readings
# Strong Positive Correlations:
# Methane and CO (0.79): These gases often increase together in transformer failures.
# Ethylene and Ethane (0.79): Both gases are closely related to thermal decomposition in transformers.
# DBDS and CO2 (0.73): This suggests that Dibenzyl Disulfide (DBDS) degradation is related to CO2 formation, likely from paper insulation breakdown.
# Health Index and Life Expectation (0.71): As expected, transformers with higher health indices tend to have longer life expectancies.
# Negative Correlations:
# Health Index vs. Acetylene (-0.52): High Acetylene concentrations strongly indicate transformer degradation, leading to a lower Health Index.
# Health Index vs. Ethylene (-0.46): High Ethylene levels suggest overheating, which negatively impacts transformer health.
# Water Content vs. Dielectric Rigidity (-0.61): More water in the transformer oil significantly reduces its dielectric strength, increasing failure risk.
# Key Insights and Next Steps
# The Health Index is highly correlated with multiple gas levels, especially Acetylene and Ethylene, which indicates thermal degradation and arcing issues.
# Methane, CO, and CO2 are closely related and should be considered in predictive modeling.
# Outliers must be analyzed carefully—removing incorrect sensor readings vs. retaining extreme values for failure prediction.
# Feature selection for the Machine Learning model should prioritize highly correlated features such as:
# Acetylene, Ethylene, Methane, CO, CO2, Water Content, and DBDS.
# By incorporating these findings, we can build a predictive model to classify transformers into Healthy, Warning, and Critical conditions, helping with predictive maintenance and failure prevention.

# In[ ]:





# 
