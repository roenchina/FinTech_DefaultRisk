# numpy and pandas for data manipulation
import numpy as np
import pandas as pd 

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File system manangement
import os

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

app_train = pd.read_csv( PARENT_DIR + '/data/application_train.csv')
print('Training data shape: ', app_train.shape)

app_test = pd.read_csv( PARENT_DIR + '/data/application_test.csv')
print('Testing data shape: ', app_test.shape)