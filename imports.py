# Standard libraries
import os
import pickle
import time

# Data manipulation and mathematical operations
import numpy as np
import pandas as pd

# Machine Learning Algorithms
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Preprocessing and model selection utilities
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Handling imbalanced datasets
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Visualization and utility tools
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

# Web app framework
import streamlit as st

# Advanced utilities
import ast
from itertools import chain

# Hyperparameter optimization
import optuna
import optuna.visualization as ov
