import sys

import pyrootutils

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=['src'],
    pythonpath=True,
)

import os
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import make_column_transformer
from sklearn.metrics import classification_report

np.seterr(divide='ignore', invalid='ignore', over='ignore')
np.random.seed(7145)


def get_cell_neighbors(center, lag, width, height):
    # Local function to test whether a location is within the bounds
    in_bounds = lambda r, c: (0 <= r < height) and (0 <= c < width)
    # Currently, (0, 0) in array style is top corner.
    # Imagine we had a regular coordinate system instead.
    # The squares form a diamond shape. By keeping track
    # of the x and y distance from the center, we can form all 4 sides at once
    # by going along just one of them.
    # We will go along the top right side, which corresponds
    # to initial positive x and y distance.
    # Ex. lag = 2 ==> (0, 2), generates (0, -2)
    # next: (1, 1), generates (-1, 1), (1, -1), and (-1, -1)
    # next: (2, 0) generates (-2, 0)
    # We'll deal with the center afterwards...
    points = []
    for x_dist in range(lag + 1):
        y_dist = lag - x_dist
        if x_dist == 0:  # Only need one other point...
            points.extend([
                (x_dist, y_dist),
                (x_dist, -y_dist)  # Bottom corner
            ])
        elif y_dist == 0:
            points.extend([
                (x_dist, y_dist),
                (-x_dist, y_dist)  # Left corner
            ])
        else:
            points.extend([
                (x_dist, y_dist),  # Top right
                (-x_dist, y_dist),  # Top left
                (x_dist, -y_dist),  # Bottom right
                (-x_dist, -y_dist)  # Bottom left
            ])
    # Filter out points that are outside of bounds...
    # And add the center while we're at it...
    # The array style coordinates is regular
    # Cartesian that's been rotated 90 degrees CW
    points = [(center[0] + r, center[1] + c) for r, c in points if in_bounds(center[0] + r, center[1] + c)]
    return tuple(zip(*points))


def show_std_err_wald(log_model: LogisticRegression, X_train: np.ndarray):
    """
    Calculates the standard errors and the Wald statistics for the coefficients
    of the logistic regression model
    :param log_model:
    :param X_train:
    :return:
    """
    # Calculate matrix of predicted class probabilities
    pred_probs = log_model.predict_proba(X_train)

    # Design matrix -- add column of 1s at the beginning of the X_train matrix
    X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

    # Initiate matrix of 0s. Fill diagonal with each predicted observation's variance.
    V = np.diagflat(np.prod(pred_probs, axis=1))
    del pred_probs

    # Covariance matrix
    # Note the @-operator, which does matrix multiplication in Python 3.5+
    cov_logit = np.linalg.inv(X_design.T @ V @ X_design)
    del X_design
    del V
    print('Covariance matrix:\n', cov_logit)

    # The standard errors are the square root of the diagonal of the covariance matrix.
    std_errs = np.sqrt(np.diag(cov_logit))
    print('Standard errors:\n', std_errs)

    # Wald statistic (coefficient / s.e.) ^ 2
    logit_params = np.insert(log_model.coef_, 0, log_model.intercept_)  # Place the intercept at the beginning
    print('Wald statistics:\n', (logit_params / std_errs) ** 2)


if __name__ == '__main__':
    LAG = 5
    metric_names = ['tavg', 'prec', 'srad', 'wind', 'vapr']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    RES = '10m'
    PREFIX = os.path.join(ROOT, 'data')
    PROCESSED_PATH = os.path.join(PREFIX, 'processed', RES)

    # Read in the fires and landcover data
    fires = np.load(os.path.join(PROCESSED_PATH, 'fires.npy'))
    landcover = np.load(os.path.join(PROCESSED_PATH, 'lc.npy'))
    # Put 0s in for all the NaNs, and convert to integer, will save some headache.
    landcover[np.isnan(landcover)] = 0
    landcover = landcover.astype(int)

    # Stack the selected metrics into a single ndarray
    metrics = np.stack([np.load(os.path.join(PROCESSED_PATH, f'{metric}.npy')) for metric in metric_names])
    # We want the months to be the first dimension, so do a slight transpose
    metrics = metrics.transpose((1, 0, 2, 3))

    landcover_cats = np.unique(landcover).tolist()[1:]  # Don't want 0
    # Save the shape, we'll do product() later :)
    N_ROWS, N_COLS = fires[0].shape
    # Create the column names...
    # We will need to reference the fire names, metric names, and
    # land cover names separately, so create a dictionary.
    fire_colnames = {}
    metric_colnames = {}
    landcover_colnames = {}

    # Create all the column names and add them to the correct dictionary
    for lag in range(1, LAG + 1):
        fire_colnames[lag] = f'fires_L{lag}_%'
        metric_colnames[lag] = [f'{metric}_L{lag}_avg' for metric in metric_names]
        landcover_colnames[lag] = [f'LC{lc}_L{lag}_%' for lc in landcover_cats]

    # Add all columns to a big list to
    # create the ultimate dataframe...
    colnames = ['Month']
    for lag in range(1, LAG + 1):
        colnames.append(fire_colnames[lag])
        colnames.extend(metric_colnames[lag])
        colnames.extend(landcover_colnames[lag])

    # Add a column for incidence of fire in the center...
    colnames.append('fireCenter')

    # The locations we can lag is determined by our
    # LARGEST lag. The list of center locations we can
    # gather stay constant throughout the process
    laggable = np.ones((N_ROWS, N_COLS), dtype=bool)
    laggable[-LAG:, :] = False
    laggable[:LAG, :] = False
    laggable[:, :LAG] = False
    laggable[:, -LAG:] = False
    # Where values aren't masked and are laggable
    fire_row_centers, fire_col_centers = np.where(~np.isnan(fires[0]) & laggable)
    # Get lagged locations for each cell...
    # at each possible lag....
    all_locs = {
        lag: [get_cell_neighbors((i, j), lag, N_COLS, N_ROWS)
              for i, j in zip(fire_row_centers, fire_col_centers)]
        for lag in range(1, LAG + 1)
    }

    # Our full dataframe
    spatial_reg = pd.DataFrame(data=None, columns=colnames)
    # For the month, all of January will be there, then February, etc.
    # The number of centers we have is how many times each month will
    # show up, so use np.repeat and fill it in right now.
    spatial_reg['Month'] = np.repeat(months, repeats=len(fire_row_centers))

    # Only process if the csv file doesn't exist...
    spatial_reg_file = os.path.join(PROCESSED_PATH, 'spatialReg.csv')
    init_start = time.perf_counter()

    if not os.path.exists(spatial_reg_file):
        start = time.perf_counter()

        for lag in range(1, LAG + 1):
            print(f'Doing lag of {lag}...', end='')
            # Extract the lag locations for each center, and reshape into a 2 x num_points array,
            # so a single indexing can be performed on our datasets.
            locs_lag = np.asarray(all_locs[lag], dtype=int)
            N_POINTS, N_LAGGED = locs_lag.shape[0], locs_lag.shape[-1]
            locs_lag = locs_lag.transpose((1, 0, 2)).reshape(2, N_POINTS * N_LAGGED)
            rows, cols = locs_lag[0].tolist(), locs_lag[1].tolist()

            # Calculate fire incidences for all the months at all the locations.
            # shape = (12, n_points, n_lagged)
            # Then, take the mean of each point for its lagged locations
            # corresponds to the last axis ==> (12, n_points).
            # Then, we assign it to the dataframe by raveling to a single column.
            # This preserves the month order we added in earlier.
            fire_incidences = fires[:, rows, cols].reshape((-1, N_POINTS, N_LAGGED))
            # Catch the "mean of empty slice" warning...
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                fire_incidences = np.nanmean(fire_incidences, axis=-1)
            spatial_reg[fire_colnames[lag]] = fire_incidences.ravel()

            # Calculate all metrics all at once using the same idea as before.
            # Extra dimension here ==> 12 x n_metrics x n_points x n_lagged.
            # Take mean on last axis, but then swap the last two axes, so that
            # the number of metrics is at the end ==> 12 x n_points x n_metrics.
            # Then flatten the first wo dimensions, and add it to the dataframe.
            collected_metrics = metrics[..., rows, cols].reshape((12, -1, N_POINTS, N_LAGGED))
            # Catch the "mean of empty slice" warning...
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                collected_metrics = np.nanmean(collected_metrics, axis=-1).transpose((0, 2, 1))
            spatial_reg[metric_colnames[lag]] = collected_metrics.reshape((-1, len(metric_names)))

            # Land cover is last.
            # We index the same way, but we have to calculate percentages of each land cover category.
            # To do this, we put the matrix into a dataframe, and then call a bunch of functions.
            #  - stack() ==> A single column dataframe with a multi-index of row then column.
            #  - groupby() ==> Group by the first level, which in this case is the 0.
            #  - value_counts() ==> Counts number of distinct values in each group i.e. row
            #  - unstack() ==> Reverses the stack operation, and puts 0 in places of missing row column pairs.
            #  - drop ==> Drops the "0" column i.e. the counts of 0, as this was a placeholder.
            lcs_df = pd.DataFrame(landcover[rows, cols].reshape((N_POINTS, N_LAGGED)), dtype=object)
            cat_counts = lcs_df.stack().groupby(level=0).value_counts().unstack(fill_value=0).drop(columns=[0])
            # Divide each row by its sum (and converts back to ndarray)
            cat_counts = cat_counts.div(cat_counts.sum(axis=1), axis=0).values
            # Right now, we have as many rows as points, so just repeat them
            # for each month we have.
            # The regular repeat will repeat in-place which isn't what we want.
            # Instead, expand into an axis, repeat on the outer axis, and get rid
            # of it afterwards.
            unwrapped = np.repeat(cat_counts[np.newaxis, ...], repeats=12, axis=0).reshape((-1, len(landcover_cats)))
            spatial_reg[landcover_colnames[lag]] = unwrapped

            print('Done!')
        end = time.perf_counter()
        print(end - start, 'SECONDS.')

        # Finally, add the fireCenter column, by raveling the fire center locations and
        # converting to ao binary-valued column.
        spatial_reg['fireCenter'] = fires[:, fire_row_centers, fire_col_centers].ravel().clip(0, 1)

        filepath = os.path.join(PROCESSED_PATH, 'spatialReg.csv')
        # Get rid of inf or -inf
        # spatial_reg[spatial_reg == float()]
        print('Saving and uploading to {}...'.format(filepath), end='')
        spatial_reg.to_csv(filepath, index=False)
        print('Done')

        end = time.perf_counter()
        print('FINISHED IN', end - init_start, 'SECONDS.')

    # Read in the regression file
    spatial_reg = pd.read_csv(spatial_reg_file).dropna()
    print(spatial_reg.shape)
    # Handle multicollinearity - remove one category column
    for lag in range(1, LAG + 1):
        spatial_reg.drop(columns=f'LC95_L{lag}_%', inplace=True)

    # Do a train test split...
    train_indices, test_indices = train_test_split(np.arange(spatial_reg.shape[0]), test_size=0.2)
    print('Train size:', len(train_indices))
    print('Test size:', len(test_indices))

    ##### LAG OF 1
    print()
    print('DOING REGRESSION')
    train_set = spatial_reg.iloc[train_indices]
    test_set = spatial_reg.iloc[test_indices]

    # Split the input from output
    X_train = train_set.iloc[:, 1:-1]
    y_train = train_set['fireCenter']
    X_test = test_set.iloc[:, 1:-1]
    y_test = test_set['fireCenter']

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)

    # Get the predictions
    predictions = logreg.predict(X_test)

    # Calculate performance with the test set
    print('UNBALANCED - NO STANDARDIZATION')
    print(classification_report(y_true=y_test, y_pred=predictions, target_names=['No Fire', 'Fire']))
    print(f'Null accuracy {(1 - np.mean(y_test)) * 100:2.3f}%', '\n')

    ######## NOW DO CLASS BALANCING AND STANDARDIZATION ########
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

    # Scaling the climatology values...
    standard_cols = [col for col in colnames if any(col.startswith(metric) for metric in ['tavg', 'srad', 'wind'])]
    power_cols = [col for col in colnames if any(col.startswith(metric) for metric in ['prec', 'vapr'])]

    transform = make_column_transformer(
        (StandardScaler(), standard_cols),
        (PowerTransformer(method='yeo-johnson'), power_cols),
        remainder='passthrough',  # To avoid dropping columns we DON'T transform
        verbose_feature_names_out=False
        # To get the final mapping of input to output columns without original transformer name.
    )
    transform.fit(X_train)

    # Transform our data and reorder columns to match the training set from earlier.
    X_transform = pd.DataFrame(data=transform.transform(X_train),
                               columns=transform.get_feature_names_out(transform.feature_names_in_))
    # Now, find the new index ordering
    col_index_ordering = [X_transform.columns.get_loc(orig_col) for orig_col in X_train.columns]
    # Reindexing into the column list with the new indices will automatically reorder them!
    X_transform = X_transform[X_transform.columns[col_index_ordering]]

    # Do the same for X_test
    X_test_transform = pd.DataFrame(data=transform.transform(X_test),
                                    columns=transform.get_feature_names_out(transform.feature_names_in_))
    X_test_transform = X_test_transform[X_test_transform.columns[col_index_ordering]]

    logreg = LogisticRegression()
    logreg.fit(X_transform, y_train)
    predictions = logreg.predict(X_test_transform)
    print('CLASS BALANCED AND STANDARDIZED')
    print(classification_report(y_true=y_test, y_pred=predictions, target_names=['No Fire', 'Fire']))
    print(f'Null accuracy {(1 - np.mean(y_test)) * 100:2.3f}%', '\n')

