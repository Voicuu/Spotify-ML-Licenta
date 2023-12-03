import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import mean_squared_error as mse, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from tabulate import tabulate
from sklearn.pipeline import make_pipeline as make_imb_pipeline
from time import time
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd
import pickle
import os
import streamlit as st