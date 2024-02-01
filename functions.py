import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support, accuracy_score


# Parse Vehicle Feature Column into usable features
def get_veh_feat_encodings(veh_feats):
    # Split String Column to individual strings for each vehicle feature
    veh_feats = veh_feats.str.split(',')

    # Make a new df based on the unique separate feature values from the step above
    veh_feats_array = veh_feats.apply(pd.Series)

    # One Hot Encode the features and return only the most preeminent ones
    veh_one_hot = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=100)
    encoding = veh_one_hot.fit_transform(veh_feats_array).toarray()

    # Return the encoder for transforming test set features, as well as the encodings for train and val sets
    return veh_one_hot, encoding


# Prepare features for ML
def get_features(feature_df, categorical_features, veh_features):
    ###################Regression Features###################
    regression_columns = feature_df.select_dtypes(exclude=['object', 'boolean'])
    scaler = MinMaxScaler()
    scaler.fit(regression_columns)

    ###################Output the Features###################
    regression_features = scaler.transform(regression_columns)
    all_features = np.concatenate((regression_features, categorical_features, veh_features), axis=1)

    # print("FEATURES:\n")
    # print(all_features)
    print(
        f"Regression Features Shape: {regression_features.shape}, Categorical Features Shape: {categorical_features.shape}, Vehicle Features Shape: {veh_features.shape}, Concatenated Shape of All Features: {all_features.shape}")
    return all_features


# Evaluate the performance of the regression models of choice
def regression_eval(df_entries, labels, ml_model, nn=False):
    percentages_array = []
    predictions_array = []

    # Predict and Print for first 10 in the provided df of features
    for index, features in enumerate(df_entries[:10]):
        features = np.expand_dims(features, 0)
        prediction = ml_model.predict(features)[0]
        if nn:
            prediction = prediction[0]
        actual = labels.iloc[index]
        error = np.abs(prediction - actual)
        error_percentage = error / actual * 100
        print(f"Prediction: ${prediction :,.0f}  Actual: ${actual :,.0f} Error: ${error :,.0f}, Error Percentage: {error_percentage :.1f}% ")

    # Predict and for all entries in the provided df of features
    # Then return the predictions and the average error percentage
    for index, features in enumerate(df_entries):
        features = np.expand_dims(features, 0)
        prediction = ml_model.predict(features)[0]
        if nn:
            prediction = prediction[0]
        predictions_array.append(prediction)
        actual = labels.iloc[index]
        error = np.abs(prediction - actual)
        error_percentage = error / actual * 100
        percentages_array.append(error_percentage)
    return predictions_array, np.mean(percentages_array)


# Plot the Predicted vs Actual values of the regression models of choice
def plot_predicted_vs_actual(predicted, actual, model_name):
    plt.Figure(figsize=(12, 8))
    # Scatter points based on the predicted and actual values of the data
    plt.scatter(actual, predicted, label=model_name, s=15)

    # Plot a Y=X Guide Line to see model accuracy
    min_val = min((min(predicted), min(actual)))
    max_val = max((max(predicted), max(actual)))
    plt.axline((min_val, min_val), (max_val, max_val), c='r')

    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()
    plt.title("Regression Plot for Validation Set ")


# Evaluate the performance of the classification models of choice
def categorical_eval(df_entries, labels, ml_model, nn=False):
    cm_predictions = []
    cm_actual = []

    # Get Actual and Predicted Categories for Val Set (NNs using One Hot, Classic ML using numerical labels)
    for index, features in enumerate(df_entries):
        features = np.expand_dims(features, 0)
        prediction = ml_model.predict(features)[0]
        if nn:
            cm_predictions.append(np.argmax(prediction))
            cm_actual.append(np.argmax(labels[index]))
        else:
            cm_predictions.append(prediction)
            cm_actual.append(labels[index])

    precision, recall, fscore, support = precision_recall_fscore_support(cm_actual, cm_predictions, average='macro',
                                                                         zero_division=1)
    accuracy = accuracy_score(cm_actual, cm_predictions)
    metrics = (accuracy, precision, recall, fscore)
    return cm_actual, cm_predictions, metrics


# Plot the confusion matrices of the classification models of choice
def plot_cm(predicted, actual, model_name):
    fig, ax = plt.subplots(figsize=(12, 8))
    # print(len(actual), len(predicted))
    ConfusionMatrixDisplay.from_predictions(actual, predicted, ax=ax, normalize='pred', cmap='inferno',
                                            xticks_rotation='vertical')
    plt.title(f"{model_name} Classification Confusion Matrix")
    plt.show()
