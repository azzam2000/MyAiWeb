import streamlit as st
import pandas as pd
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Define the Streamlit app
def app():
    st.title('LazyPredict Framework')

    # Upload the CSV file
    st.write('### Upload CSV File')
    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Select the model type
        st.write('### Model Selection')
        model_type = st.selectbox('Select a model type:', ['Predictive', 'Classification'])

        if model_type == 'Predictive':
            # Show the data
            st.write('### Dataset')
            st.write(df.head())

            # Get the target column
            target_col = st.selectbox('Select a target column:', options=df.columns)

            # Get the feature columns
            feature_cols = list(df.columns)
            feature_cols.remove(target_col)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=42)

            # Instantiate the LazyRegressor model
            reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

            # Fit the model on the training data
            models, predictions = reg.fit(X_train, X_test, y_train, y_test)

            # Show the results
            st.write('### Results')
            st.write(models)

            # Visualize the results
            fig, ax = plt.subplots()
            ax.barh(models.index, models['R-Squared'])
            ax.set_xlabel('R-Squared')
            ax.set_ylabel('Model')
            ax.set_title('Comparison of Regression Models')
            st.pyplot(fig)

            # Save the final model
            if st.button('Save Model'):
                joblib.dump(models, 'regression_model.joblib')
                st.write('Model saved successfully!')

        elif model_type == 'Classification':
            # Show the data
            st.write('### Dataset')
            st.write(df.head())

            # Get the target column
            target_col = st.selectbox('Select a target column:', options=df.columns)

            # Get the feature columns
            feature_cols = list(df.columns)
            feature_cols.remove(target_col)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_col], test_size=0.3, random_state=42)

            # Instantiate the LazyClassifier model
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

            # Fit the model on the training data
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)

            # Show the results
            st.write('### Results')
            st.write(models)

            # Visualize the results
            fig, ax = plt.subplots()
            ax.barh(models.index, models['Accuracy'])
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Model')
            ax.set_title('Comparison of Classification Models')
            st.pyplot(fig)

            # Save the final model
            if st.button('Save Model'):
                joblib.dump(models, 'classification_model.joblib')
                st.write('Model saved successfully!')

    # Load a saved model and allow the user to
# Load a saved model and allow the user to upload a new dataset to test the model
def test_model():
    st.title('Test the Model')

    # Load the saved model
    model_file = st.file_uploader('Upload a saved model', type='joblib')
    if model_file is not None:
        model = joblib.load(model_file)

        # Upload the CSV file
        st.write('### Upload CSV File')
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Show the data
            st.write('### Dataset')
            st.write(df.head())

            # Get the target column
            target_col = st.selectbox('Select a target column:', options=df.columns)

            # Get the feature columns
            feature_cols = list(df.columns)
            feature_cols.remove(target_col)

            # Make predictions on the new data
            predictions = model.predict(df[feature_cols])

            # Show the predictions
            st.write('### Predictions')
            st.write(predictions)
            #
if __name__ == '__main__':
    app()