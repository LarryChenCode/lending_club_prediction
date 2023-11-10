import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# Data Dictionary
df_dict = pd.read_csv("../data/raw/data_dictionary.csv")
df_dict

# Load Data
df = pd.read_csv("../data/raw/lending_club_2007_2011_6_states.csv")
df.head()

# Check the info and missing value of the data
df.info(verbose=True)
df.isnull().sum()

# Summary Statistics
df.describe()
