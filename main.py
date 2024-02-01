from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import time

from functions import *

############################# Model Flags #############################
"""
To see the process of any specific model, change it's flag value below to binary True. 
I have selected two Random Forest models for my final predictions, so it currently has the only flag set to True.
"""
# Random Forest
RF_flag = True
# K-Nearest Neighbors
KNN_flag = False
# Naive Bayes
NB_flag = False
# Deep Neural Network
DNN_flag = False
# Wide Neural Network
WNN_flag = False

############################# Load and Prepare Data #############################
if __name__ == "__main__":
    # Load Train CSV File and check shape
    # Drop any entries with a blank column
    df = pd.read_csv('./Training_DataSet.csv').set_index('ListingID').sample(frac=1).ffill()
    print(f"Original df Shape: {df.shape}")

    # Get Columns to make Features out of
    useful_feature_columns = ['SellerCity',
                              # 'SellerIsPriv',
                              # 'SellerListSrc',
                              # 'SellerName',
                              'SellerRating',
                              # 'SellerRevCnt',
                              'SellerState',
                              # 'SellerZip',
                              'VehBodystyle',
                              'VehCertified',
                              'VehColorExt',
                              'VehColorInt',
                              'VehDriveTrain',
                              'VehEngine',
                              # 'VehFeats',
                              'VehFuel',
                              'VehHistory',
                              'VehListdays',
                              'VehMake',
                              'VehMileage',
                              'VehModel',
                              'VehPriceLabel',
                              # 'VehSellerNotes',
                              'VehType',
                              'VehTransmission',
                              'VehYear',
                              'Vehicle_Trim', 'Dealer_Listing_Price'  # Labels
                              ]

    # Pull out Veh Feats aside for string parsing
    veh_feats = df['VehFeats']

    # Turn vehicle year into a categorical feature and not regression feature
    df['VehYear'] = df['VehYear'].astype(str)

    df = df[useful_feature_columns]
    print(f"Filtered df shape: {df.shape}")
    # One Hot Encode String and Boolean Categorical Columns
    feat_enc = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=50)
    categorical_columns = df.select_dtypes(include=['object', 'boolean']).iloc[:, :-1]
    categorical_features = feat_enc.fit_transform(categorical_columns).toarray()

    # Split df to have 85% of data for training and 15% for validation
    train_split = 0.85
    train_samples = int(len(df)*train_split)

    train_df = df[:train_samples]
    train_cat_features = categorical_features[:train_samples]
    val_df = df[train_samples:]
    val_cat_features = categorical_features[train_samples:]

    print(f"train_df shape: {train_df.shape}")
    print(f"val_df shape: {val_df.shape}")

    # Encode Categorical Labels (One Hot for NNs and Numerical for Classical ML)
    oh_lab_enc = OneHotEncoder()
    le_lab_enc = LabelEncoder()

    categorical_labels_one_hot = oh_lab_enc.fit_transform(df['Vehicle_Trim'].to_frame()).toarray()
    categorical_labels_enc = le_lab_enc.fit_transform(df['Vehicle_Trim'].to_frame().squeeze())

    regression_labels = df['Dealer_Listing_Price']

    # Train and Validation Split
    train_categorical_labels_one_hot = categorical_labels_one_hot[:train_samples]
    train_categorical_labels_enc = categorical_labels_enc[:train_samples]
    train_regression_labels = regression_labels[:train_samples]

    val_categorical_labels_one_hot = categorical_labels_one_hot[train_samples:]
    val_categorical_labels_enc = categorical_labels_enc[train_samples:]
    val_regression_labels = regression_labels[train_samples:]

    # Filter out Labels from split dfs for regression feature encoding
    train_df = train_df.loc[:, ~train_df.columns.isin(['Vehicle_Trim', 'Dealer_Listing_Price'])]
    val_df = val_df.loc[:, ~val_df.columns.isin(['Vehicle_Trim', 'Dealer_Listing_Price'])]

    # View a portion of the df
    # print(val_df.head())

    veh_one_hot, veh_feat_encoding = get_veh_feat_encodings(veh_feats)
    print(f"Shape of Vehicle Feature Encodings for Train and Val Sets: {veh_feat_encoding.shape}")
    veh_feat_encoding_train = veh_feat_encoding[:train_samples]
    veh_feat_encoding_val = veh_feat_encoding[train_samples:]

    # Prepare features for ML
    train_features = get_features(train_df, train_cat_features, veh_feat_encoding_train)

    ############################# Testing ML Models (Regression) #############################

    ######### Random Forest Regression #########
    if RF_flag:
        time_now = time.time()

        random_forest_regression = RandomForestRegressor()
        random_forest_regression.fit(train_features, train_regression_labels)

        print(f"Random Forest Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### K-Nearest Neighbors Regression #########
    if KNN_flag:
        time_now = time.time()

        knn_regression = KNeighborsRegressor(n_neighbors=4)
        knn_regression.fit(train_features, train_regression_labels)

        print(f"K Nearest Neighbors Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### Naive Bayes Regression #########
    if NB_flag:
        time_now = time.time()

        naive_bayes_regression = GaussianNB()
        naive_bayes_regression.fit(train_features, train_regression_labels)

        print(f"Naive Bayes Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### Deep Regression Neural Network #########
    if DNN_flag:
        time_now = time.time()

        deep_neural_network_regression = Sequential([
                                    Dense(2048, input_shape=[train_features.shape[1]], activation='relu'),
                                    Dropout(0.30),

                                    Dense(1024, activation='relu'),

                                    Dense(512, activation='relu'),

                                    Dense(256, activation='relu'),

                                    Dense(128, activation='relu'),

                                    Dense(32, activation='relu'),

                                    Dense(1)
                                    ])

        deep_neural_network_regression.compile(optimizer='adam', loss='huber', metrics='mae')
        deep_neural_network_regression.fit(train_features, train_regression_labels, batch_size=256, epochs=50, verbose=1)

        print(f"Deep Neural Net Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### Wide Regression Neural Network #########
    if WNN_flag:
        time_now = time.time()

        wide_neural_network_regression = Sequential([
                                    Dense(train_features.shape[1], input_shape=[train_features.shape[1]], activation='relu'),
                                    Dense(1)
                                    ])

        wide_neural_network_regression.compile(optimizer='adam', loss='huber', metrics='mae')
        wide_neural_network_regression.fit(train_features, train_regression_labels, batch_size=256, epochs=50, verbose=1)

        print(f"Wide Neural Net Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ############################# REGRESSION VALIDATION SET RESULTS #############################
    print("Validation Set:")
    val_features = get_features(val_df, val_cat_features, veh_feat_encoding_val)

    if RF_flag:
        print("\nRANDOM FOREST:")
        random_forest_predictions, random_forest_error = regression_eval(val_features, val_regression_labels, random_forest_regression)

    if KNN_flag:
        print("\nK NEAREST NEIGHBORS:")
        knn_predictions, knn_error= regression_eval(val_features, val_regression_labels, knn_regression)

    if NB_flag:
        print("\nNAIVE BAYES:")
        naive_bayes_predictions, naive_bayes_error = regression_eval(val_features, val_regression_labels, naive_bayes_regression)

    if DNN_flag:
        print("\nDEEP NEURAL NETWORK:")
        deep_neural_network_predictions, deep_neural_network_error = regression_eval(val_features, val_regression_labels, deep_neural_network_regression, nn=True)

    if WNN_flag:
        print("\nWIDE NEURAL NETWORK:")
        wide_neural_network_predictions, wide_neural_network_error = regression_eval(val_features, val_regression_labels, wide_neural_network_regression, nn=True)

    if RF_flag:
        print(f"Mean Error Percentage for Random Forest: {random_forest_error :,.0f}%")
    if KNN_flag:
        print(f"Mean Error Percentage for K Nearest Neighbors: {knn_error :,.0f}%")
    if NB_flag:
        print(f"Mean Error Percentage for Naive Bayes: {naive_bayes_error :,.0f}%")
    if DNN_flag:
        print(f"Mean Error Percentage for Deep Neural Network: {deep_neural_network_error :,.0f}%")
    if WNN_flag:
        print(f"Mean Error Percentage for Wide Neural Network: {wide_neural_network_error :,.0f}%")

    ############################# Plot Results #############################
    if RF_flag:
        plot_predicted_vs_actual(random_forest_predictions, val_regression_labels, "Random Forest")

    if KNN_flag:
        plot_predicted_vs_actual(knn_predictions, val_regression_labels, "KNN")

    if NB_flag:
        plot_predicted_vs_actual(naive_bayes_predictions, val_regression_labels, "Naive Bayes")

    if DNN_flag:
        plot_predicted_vs_actual(deep_neural_network_predictions, val_regression_labels, "Deep Neural Network")

    if WNN_flag:
        plot_predicted_vs_actual(wide_neural_network_predictions, val_regression_labels, "Wide Neural Network")

    print('Regression Confusion Matrix Plot: If every prediction is correct, every scatter point will sit on the y=x red line')

    ############################# Testing ML Models (Classification) #############################

    ######### Random Forest Classifier #########
    if RF_flag:
        time_now = time.time()

        random_forest_classification = RandomForestClassifier()
        random_forest_classification.fit(train_features, train_categorical_labels_enc)

        print(f"Random Forest Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### K-Nearest Neighbors Classifier #########
    if KNN_flag:
        time_now = time.time()

        knn_classification = KNeighborsClassifier(n_neighbors=4)
        knn_classification.fit(train_features, train_categorical_labels_enc)

        print(f"K Nearest Neighbors Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    #########  Naive Bayes Classifier #########
    if NB_flag:
        time_now = time.time()

        naive_bayes_classification = MultinomialNB()
        naive_bayes_classification.fit(train_features, train_categorical_labels_enc)

        print(f"Naive Bayes Train Time: {(time.time()-time_now)/60 :.2f} minutes")

    ######### Deep Classification Neural Network #########
    if DNN_flag:
        time_now = time.time()
        deep_neural_network_classification = Sequential([
                                    Dense(2048, input_shape=[train_features.shape[1]], activation='relu'),
                                    Dropout(0.15),

                                    Dense(1024, activation='relu'),

                                    Dense(512, activation='relu'),

                                    Dense(256, activation='relu'),

                                    Dense(128, activation='relu'),

                                    Dense(29, activation='softmax')
                                    ])
        deep_neural_network_classification.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics='accuracy')
        deep_neural_network_classification.fit(train_features, train_categorical_labels_one_hot, batch_size=256, epochs=50, verbose=1)
        print(f"Deep Neural Net Train Time: {(time.time()-time_now)/60 :.2f} minutes")

        deep_neural_network_classification.evaluate(val_features, val_categorical_labels_one_hot)

    ######### Wide Classification Network #########
    if WNN_flag:
        time_now = time.time()
        wide_neural_network_classification = Sequential([
                                Dense(train_features.shape[1], input_shape=[train_features.shape[1]], activation='relu'),
                                Dense(29, activation='softmax')
                                ])

        wide_neural_network_classification.compile(optimizer='adam', loss='CategoricalCrossentropy', metrics='accuracy')
        wide_neural_network_classification.fit(train_features, train_categorical_labels_one_hot, batch_size=256, epochs=50, verbose=1)
        print(f"Wide Neural Net Train Time: {(time.time()-time_now)/60 :.2f} minutes")

        wide_neural_network_classification.evaluate(val_features, val_categorical_labels_one_hot)

    ############################# CLASSIFICATION VALIDATION SET RESULTS #############################
    print("Validation Set:")
    val_features = get_features(val_df, val_cat_features, veh_feat_encoding_val)

    if RF_flag:
        print("\nRANDOM FOREST:")
        cm_actual, random_forest_predictions, random_forest_metrics = categorical_eval(val_features, val_categorical_labels_enc, random_forest_classification)
    if KNN_flag:
        print("\nK NEAREST NEIGHBORS:")
        _, knn_predictions, knn_metrics = categorical_eval(val_features, val_categorical_labels_enc, knn_classification)
    if NB_flag:
        print("\nNAIVE BAYES:")
        _, naive_bayes_predictions, naive_bayes_metrics = categorical_eval(val_features, val_categorical_labels_enc, naive_bayes_classification)
    if DNN_flag:
        print("\nDEEP NEURAL NETWORK:")
        _, deep_neural_network_predictions, deep_neural_network_metrics = categorical_eval(val_features, val_categorical_labels_one_hot, deep_neural_network_classification, nn=True)
    if WNN_flag:
        print("\nWIDE NEURAL NETWORK:")
        _, wide_neural_network_predictions, wide_neural_network_metrics = categorical_eval(val_features, val_categorical_labels_one_hot, wide_neural_network_classification, nn=True)

    if RF_flag:
        print(f"Random Forest Categorical Prediction Metrics (accuracy, precision, recall, fscore): {random_forest_metrics}")
    if KNN_flag:
        print(f"K Nearest Neighbors Categorical Prediction Metrics (accuracy, precision, recall, fscore): {knn_metrics}")
    if NB_flag:
        print(f"Naive Bayes Categorical Prediction Metrics (accuracy, precision, recall, fscore): {naive_bayes_metrics}")
    if DNN_flag:
        print(f"Deep Neural Network Categorical Prediction Metrics (accuracy, precision, recall, fscore): {deep_neural_network_metrics}")
    if WNN_flag:
        print(f"Wide Neural Network Categorical Prediction Metrics (accuracy, precision, recall, fscore): {wide_neural_network_metrics}")

    ############################# Plot Results #############################

    # Plot the 4 Confusion Matrix Metrics for each model
    fig, ax = plt.subplots(figsize=(12, 8))

    metrics_labels = ['accuracy', 'precision', 'recall', 'fscore']
    bar_width = 0.12
    bar_shift = 0
    index = np.arange(len(metrics_labels))

    # Plot bars for each model next to each other for each metric
    if RF_flag:
        ax.bar(index, random_forest_metrics, bar_width, label='Random Forest')
        bar_shift += bar_width
    if KNN_flag:
        ax.bar(index + bar_shift, knn_metrics, bar_width, label='KNN')
        bar_shift += bar_width
    if NB_flag:
        ax.bar(index + bar_shift, naive_bayes_metrics, bar_width, label='Naive Bayes')
        bar_shift += bar_width
    if DNN_flag:
        ax.bar(index + bar_shift, deep_neural_network_metrics, bar_width, label='Deep NN')
        bar_shift += bar_width
    if WNN_flag:
        ax.bar(index + bar_shift, wide_neural_network_metrics, bar_width, label='Wide NN')

    # Center X-axis ticks
    if bar_shift == bar_width:
        ax.set_xticks(index)
    elif bar_shift % bar_width == 0:
        ax.set_xticks(index + bar_shift/2)
    else:
        ax.set_xticks(index + (bar_shift-bar_width)/2)

    ax.set_xticklabels(metrics_labels)
    ax.legend()
    plt.title("Classification Metrics for Validation Set")
    plt.show()

    # Plot Confusion Matrices for each model
    print('Classification Confusion Matrix:')
    if RF_flag:
        plot_cm(random_forest_predictions, cm_actual, "Random Forest")
    if KNN_flag:
        plot_cm(knn_predictions, cm_actual, "KNN")
    if NB_flag:
        plot_cm(naive_bayes_predictions, cm_actual, "Naive Bayes")
    if DNN_flag:
        plot_cm(deep_neural_network_predictions, cm_actual, "Deep Neural Network")
    if WNN_flag:
        plot_cm(wide_neural_network_predictions, cm_actual, "Wide Neural Network")

    ############################# Plot for Final Model Decision #############################

    # Plot 1 minus Regression Prediction Error and Classification Accuracy for each model

    fig, ax = plt.subplots(figsize=(12, 8))

    metrics_labels = ['regression', 'classification']
    bar_width = 0.12
    bar_shift = 0
    index = np.arange(2)

    # Plot metrics for each model
    if RF_flag:
        ax.bar(index, (1-random_forest_error/100, random_forest_metrics[0]), bar_width, label='Random Forest')
        bar_shift += bar_width
    if KNN_flag:
        ax.bar(index + bar_shift, (1-knn_error/100, knn_metrics[0]), bar_width, label='KNN')
        bar_shift += bar_width
    if NB_flag:
        ax.bar(index + bar_shift, (1-naive_bayes_error/100, naive_bayes_metrics[0]), bar_width, label='Naive Bayes')
        bar_shift += bar_width
    if DNN_flag:
        ax.bar(index + bar_shift, (1-deep_neural_network_error/100, deep_neural_network_metrics[0]), bar_width, label='Deep NN')
        bar_shift += bar_width
    if WNN_flag:
        ax.bar(index + bar_shift, (1-wide_neural_network_error/100, wide_neural_network_metrics[0]), bar_width, label='Wide NN')

    # Center X-axis ticks
    if bar_shift == bar_width:
        ax.set_xticks(index)
    elif bar_shift % bar_width == 0:
        ax.set_xticks(index + bar_shift/2)
    else:
        ax.set_xticks(index + (bar_shift-bar_width)/2)
    ax.set_xticklabels(metrics_labels)
    ax.legend()
    plt.title("Regression and Classification Performance comparison for all tested models")
    plt.show()

    ############################# MODEL CHOICE: RANDOM FOREST #############################

    test_df = pd.read_csv('./Test_Dataset.csv').set_index('ListingID').ffill()

    # Pull Out Vehicle Features
    test_veh_feats = test_df['VehFeats']

    # Make a new df based on the unique separate feature values from the step above
    veh_feats_array = test_veh_feats.str.split(',').apply(pd.Series)
    veh_feats_array[len(veh_feats_array.columns)+1] = np.zeros

    # Transform Vehicle Features based on One Hot Encoder from Train and Val set
    test_veh_feats = veh_one_hot.transform(veh_feats_array).toarray()

    # Filter df into only useful features, not including last 2 columns as they are the label column names
    test_feature_columns = useful_feature_columns[:-2]
    test_df = test_df[test_feature_columns]

    # Turn vehicle year into a categorical feature and not regression feature
    test_df['VehYear'] = test_df['VehYear'].astype(str)

    # One Hot Encode String and Boolean Categorical Columns
    categorical_columns = test_df.select_dtypes(include=['object', 'boolean'])
    categorical_features = feat_enc.transform(categorical_columns).toarray()

    test_features = get_features(test_df, categorical_features, test_veh_feats)

    # Output test data predictions to a new df
    output_df = pd.DataFrame()
    output_df.index = test_df.index

    output_df['Vehicle_Trim'] = le_lab_enc.inverse_transform(random_forest_classification.predict(test_features))
    output_df['Dealer_Listing_Price'] = [price for price in random_forest_regression.predict(test_features)]
    output_df.to_csv('./output.csv')
