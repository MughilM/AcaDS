import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['src'],
    pythonpath=True,
)

import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report

import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential

np.random.seed(7145)


if __name__ == '__main__':
    LAG = 5
    metric_names = ['tavg', 'prec', 'srad', 'wind', 'vapr']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    RES = '10m'
    PREFIX = os.path.join(ROOT, 'data')
    PROCESSED_PATH = os.path.join(PREFIX, 'processed', RES)

    spatial_reg = pd.read_csv(os.path.join(PROCESSED_PATH, 'spatialReg.csv'))

    # We'll do the same processing that led to the improved performance on the logistic regression,
    # namely, class balancing and standardization on the climatology values.
    # First handle the multicollinearity.
    for lag in range(1, LAG + 1):
        spatial_reg.drop(columns=f'LC95_L{lag}_%', inplace=True)
    # Get rid of rows with NaNs in them...
    spatial_reg = spatial_reg.dropna()
    # Do a train test split
    train_indices, test_indices = train_test_split(np.arange(spatial_reg.shape[0]), test_size=0.2)
    train_set = spatial_reg.iloc[train_indices]
    test_set = spatial_reg.iloc[test_indices]

    # Split the input from output
    X_train = train_set.iloc[:, 1:-1]
    y_train = train_set['fireCenter']
    X_test = test_set.iloc[:, 1:-1]
    y_test = test_set['fireCenter']

    # Equalize the number of positive and negative samples in the training set...
    no_fire_samples = train_set[train_set['fireCenter'] == 0]
    fire_samples = train_set[train_set['fireCenter'] == 1]
    # Randomly choose the number of 1 samples we have from the no fire samples
    chosen_no_fire_samples = no_fire_samples.sample(n=fire_samples.shape[0])
    # Concatenate both sets together, and shuffle with .sample(frac=1)
    train_set = pd.concat((chosen_no_fire_samples, fire_samples), axis=0, ignore_index=True).sample(frac=1)
    print('New number of records:', train_set.shape[0])
    print('New proportion of fires:', np.mean(train_set['fireCenter']))

    # Split off X and y train again
    X_train = train_set.iloc[:, 1:-1]
    y_train = train_set['fireCenter']

    standard_cols = [col for col in spatial_reg.columns if any(col.startswith(metric)
                                                               for metric in ['tavg', 'srad', 'wind'])]
    power_cols = [col for col in spatial_reg.columns if any(col.startswith(metric)
                                                            for metric in ['prec', 'vapr'])]

    transform = make_column_transformer(
        (StandardScaler(), standard_cols),
        (PowerTransformer(method='yeo-johnson'), power_cols),
        remainder='passthrough',  # To avoid dropping columns we DON'T transform
        verbose_feature_names_out=False
        # To get the final mapping of input to output columns without original transformer name.
    )
    transform.fit(X_train)

    # Create a transformed DataFrame, with the transformed data, and the new column ordering
    X_transform = pd.DataFrame(data=transform.transform(X_train),
                               columns=transform.get_feature_names_out(transform.feature_names_in_))
    # Now, find the new index ordering
    col_index_ordering = [X_transform.columns.get_loc(orig_col) for orig_col in X_train.columns]
    # Reindexing into the column list with the new indices will automatically reorder them!
    X_transform = X_transform[X_transform.columns[col_index_ordering]]

    X_test_transform = pd.DataFrame(data=transform.transform(X_test),
                                    columns=transform.get_feature_names_out(transform.feature_names_in_))
    X_test_transform = X_test_transform[X_test_transform.columns[col_index_ordering]]

    # Build the model, there are 105 features in the full dataframe
    in_features = X_transform.shape[1]

    model = Sequential()
    model.add(layers.Dense(75, activation='relu', input_shape=(in_features,)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output!

    print(model.summary())

    # For fitting, we will use the Adam optimizer and binary cross entropy
    print('Fitting...')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.fit(X_transform, y_train, batch_size=64, epochs=30, validation_data=(X_test_transform, y_test))

    # Get the predictions
    predictions = model.predict(X_test_transform).ravel()
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1

    print(classification_report(y_true=y_test, y_pred=predictions))
    print(f'Null accuracy {(1 - np.mean(y_test)) * 100:2.3f}%', '\n')



