import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA, TruncatedSVD
from sklearn.datasets import load_iris, load_diabetes, load_wine


@st.cache_data
def get_data_heart_disease():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'heart_statlog.csv'))
    df.loc[df['chest pain type'] == 1, 'chest pain type'] = 'typical angina'
    df.loc[df['chest pain type'] == 2, 'chest pain type'] = 'atypical angina'
    df.loc[df['chest pain type'] == 3, 'chest pain type'] = 'non-anginal pain'
    df.loc[df['chest pain type'] == 4, 'chest pain type'] = 'asymptomatic'
    df['chest pain type'] = df['chest pain type'].astype(str)

    df.loc[df['sex'] == 1, 'sex'] = 'male'
    df.loc[df['sex'] == 0, 'sex'] = 'female'
    df['sex'] = df['sex'].astype(str)

    df.loc[df['resting ecg'] == 0, 'resting ecg'] = 'normal'
    df.loc[df['resting ecg'] == 1, 'resting ecg'] = 'ST-T wave abnormality'
    df.loc[df['resting ecg'] == 2, 'resting ecg'] = 'probable or definite left ventricular hypertrophy'
    df['resting ecg'] = df['resting ecg'].astype(str)

    df.loc[df['exercise angina'] == 0, 'exercise angina'] = 'no'
    df.loc[df['exercise angina'] == 1, 'exercise angina'] = 'yes'
    df['exercise angina'] = df['exercise angina'].astype(str)

    df.loc[df['ST slope'] == 0, 'ST slope'] = 'unsloping'
    df.loc[df['ST slope'] == 1, 'ST slope'] = 'flat'
    df.loc[df['ST slope'] == 2, 'ST slope'] = 'downslopping'
    df['ST slope'] = df['ST slope'].astype(str)
    return df


def get_data_titanic():
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'titanic.csv'))
    target = df.pop("Survived")
    df.insert(len(df.columns), "Survived", target)
    return df


def get_dim_reduc_algo(algorithm, hyperparameters):
    if algorithm == 'None':
        return 'passthrough'
    if algorithm == 'PCA':
        return PCA(n_components=hyperparameters['n_components'])
    if algorithm == 'LDA':
        return LDA(solver=hyperparameters['solver'])
    if algorithm == 'Kernel PCA':
        return KernelPCA(n_components=hyperparameters['n_components'], kernel=hyperparameters['kernel'])
    if algorithm == 'Truncated SVD':
        return TruncatedSVD(n_components=hyperparameters['n_components'])


def split_columns(df, drop_cols=[]):
    # numerical columns
    num_cols_extracted = [col for col in df.select_dtypes(include='number').columns if col not in drop_cols]
    num_cols = []
    num_cols_missing = []
    cat_cols = []
    cat_cols_missing = []
    for col in num_cols_extracted:
        if (len(df[col].unique()) < 15):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    # categorical columns
    obj_cols = [col for col in df.select_dtypes(exclude=['number']).columns if col not in drop_cols]
    text_cols = []
    for col in obj_cols:
        if (len(df[col].unique()) < 25):
            cat_cols.append(col)
        else:
            text_cols.append(col)

    return num_cols, cat_cols, text_cols, num_cols_missing, cat_cols_missing


def load_dataset(dataset):
    if (dataset == 'Load my own dataset'):
        uploaded_file = st.file_uploader('File uploader')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    elif (dataset == 'Titanic dataset'):
        df = get_data_titanic()
    elif (dataset == 'Heart disease dataset'):
        df = get_data_heart_disease()
    elif (dataset == 'Iris dataset'):
        df = load_iris(as_frame=True).data
        df["target"] = load_iris(as_frame=True).target
    elif (dataset == 'Diabetes dataset'):
        df = load_diabetes(as_frame=True).data
        df["target"] = load_diabetes(as_frame=True).target
    elif (dataset == 'Wine dataset'):
        df = load_wine(as_frame=True).data
        df["target"] = load_wine(as_frame=True).target
    return df


def wrapper_selectbox(label, options, visible=True):
    if (not visible):
        return 'None'
    return st.sidebar.selectbox(label, options)


# configuration of the page
st.set_page_config(layout="wide")

SPACER = .2
ROW = 1

title_spacer1, title, title_spacer_2 = st.columns((.1, ROW, .1))
with title:
    st.title('High-dimensional visualization tool')
