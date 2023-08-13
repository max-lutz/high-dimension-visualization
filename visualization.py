import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
        return LinearDiscriminantAnalysis(solver=hyperparameters['solver'], n_components=hyperparameters['n_components'])
    if algorithm == 'Kernel PCA':
        return KernelPCA(n_components=hyperparameters['n_components'], kernel=hyperparameters['kernel'])
    if algorithm == 'Truncated SVD':
        return TruncatedSVD(n_components=hyperparameters['n_components'], algorithm=hyperparameters['solver'])


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
    df = None
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

title_spacer2, subtitle, title_spacer_2 = st.columns((0.16, ROW, 0.7))
with subtitle:
    st.write("""
            A tool to display high dimensional datasets as 2D and 3D graphs. 
            Test the app with the base datasets or upload your own.
            The app automatically preprocess the dataset into a high-dimensional dataset.
            You can then choose the way to reduce the dimensions and visualize the result.
            """)
    st.write("")


st.write("")
dataset = st.selectbox('Select dataset', ['Titanic dataset', 'Heart disease dataset', 'Iris dataset',
                                          'Wine dataset', 'Load my own dataset'])
df = load_dataset(dataset)

if (df is not None):
    st.sidebar.header('Select feature to predict')
    _, cat_cols, _, _, _ = split_columns(df)
    target_list = [x for x in df.columns.to_list() if x in cat_cols]
    target_list.reverse()
    target_selected = st.sidebar.selectbox('Predict', target_list)

    X = df.drop(columns=target_selected)
    Y = df[target_selected].values.ravel()

    drop_cols = []
    num_cols, cat_cols, text_cols, num_cols_missing, cat_cols_missing = split_columns(X, drop_cols)

    # create new lists for columns with missing elements
    for col in X.columns:
        if (col in num_cols and X[col].isna().sum() > 0):
            num_cols.remove(col)
            num_cols_missing.append(col)
        if (col in cat_cols and X[col].isna().sum() > 0):
            cat_cols.remove(col)
            cat_cols_missing.append(col)

    # combine text columns in one new column because countVectorizer does not accept multiple columns
    text_cols_original = text_cols
    if (len(text_cols) != 0):
        X['text'] = X[text_cols].astype(str).agg(' '.join, axis=1)
        for cols in text_cols:
            drop_cols.append(cols)
        text_cols = "text"

    missing_cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent', missing_values=np.nan))
    missing_cat_pipeline.steps.append(('encoding', OneHotEncoder(handle_unknown='ignore')))

    missing_num_pipeline = make_pipeline(SimpleImputer(strategy='median', missing_values=np.nan))
    missing_num_pipeline.steps.append(('scaling', StandardScaler()))

    # need to make two preprocessing pipeline too handle the case encoding without imputer...
    preprocessing = make_column_transformer(
        (missing_cat_pipeline, cat_cols_missing),
        (missing_num_pipeline, num_cols_missing),

        (OneHotEncoder(handle_unknown='ignore'), cat_cols),
        (CountVectorizer(), text_cols),
        (StandardScaler(), num_cols)
    )

    st.header('Preprocessing')
    X_preprocessed = preprocessing.fit_transform(X)
    row1_spacer1, row1_1, row1_spacer2, row1_2, row1_spacer3 = st.columns((SPACER/10, ROW, SPACER, ROW, SPACER/10))
    with row1_1:
        st.header('Original dataset')
        st.text(f'Original dataframe has shape: {X.shape}')
        st.write(df)

    with row1_2:

        st.header('Preprocessed dataset')
        display_dataframe = True
        try:
            pd.DataFrame(X_preprocessed)
        except:
            display_dataframe = False
        if (X_preprocessed.shape[1] < 100 and display_dataframe):
            st.text(f'Processed dataframe has shape: {X_preprocessed.shape}')
            st.write(X_preprocessed)
        else:
            st.text(f'Processed dataframe cannot be displayed, shape: {X_preprocessed.shape}')

    dim = preprocessing.fit_transform(X).shape[1]
    if (dim > 2):
        st.sidebar.title('Dimension reduction')
        dimension_reduction_algorithm = st.sidebar.selectbox('Algorithm', ['Kernel PCA', 'Truncated SVD'])

        hyperparameters_dim_reduc = {}
        hyperparameters_dim_reduc['n_components'] = 2
        if (dimension_reduction_algorithm == 'Kernel PCA'):
            hyperparameters_dim_reduc['kernel'] = st.sidebar.selectbox(
                'Kernel (default = linear)', ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'])
        if (dimension_reduction_algorithm == 'Truncated SVD'):
            hyperparameters_dim_reduc['solver'] = st.sidebar.selectbox(
                'Solver (default = randomized)', ['randomized', 'arpack'])
    else:
        st.sidebar.title('Dimension reduction')
        dimension_reduction_algorithm = st.sidebar.selectbox('Number of features too low', ['None'])
        hyperparameters_dim_reduc = {}

    dimension_reduction_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        ('dimension reduction', get_dim_reduc_algo(dimension_reduction_algorithm, hyperparameters_dim_reduc))
    ])

    dimension_reduction_pipeline.fit(X, Y)
    X_reduced = pd.DataFrame(dimension_reduction_pipeline.transform(X))
    X_reduced.columns = [f"component {i+1}" for i in range(len(X_reduced.columns))]
    X_reduced['target'] = Y

    if (len(X_reduced.columns) == 3):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title('Dataset with reduced dimensions')
        sns.scatterplot(x='component 1', y='component 2', data=X_reduced, hue='target', ax=ax)
        st.pyplot(fig)

    # if (len(X_reduced.columns) == 4):
    #     fig = px.scatter_3d(X_reduced, x='component 1', y='component 2', z='component 3',
    #                         symbol='target', color='target', width=800, height=400)
    #     st.plotly_chart(fig, use_container_width=True)

else:
    st.sidebar.header('')
