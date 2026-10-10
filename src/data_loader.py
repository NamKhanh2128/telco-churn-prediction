'''Data loading and validation for the IBM Telco churn dataset.'''

from hashlib import sha256

import pandas as pd

from src.config import (
    DATASET_PATH,
    DROP_TOTAL_CHARGES,
    ID_COL,
    NUMERIC_FEATURES,
    TARGET_COL,
)


REQUIRED_COLUMNS = {
    ID_COL,
    TARGET_COL,
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
}


def get_file_sha256(path):
    '''Return a reproducibility hash for a dataset or model artifact.'''
    digest = sha256()
    with open(path, 'rb') as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_schema(df):
    '''Validate the minimum IBM Telco schema before transformations.'''
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    target_values = set(df[TARGET_COL].dropna().astype(str).str.strip().unique())
    unexpected = target_values.difference({'Yes', 'No', '0', '1'})
    if unexpected:
        raise ValueError(f'Unexpected target values: {sorted(unexpected)}')

    duplicate_ids = int(df[ID_COL].duplicated().sum())
    duplicate_rows = int(df.duplicated().sum())
    return {
        'rows': int(len(df)),
        'columns': int(df.shape[1]),
        'duplicate_ids': duplicate_ids,
        'duplicate_rows': duplicate_rows,
        'missing_values': {
            str(column): int(value)
            for column, value in df.isna().sum().items()
            if value
        },
    }


def load_raw_data(path=None, drop_total_charges=None):
    '''Load, minimally clean and validate the IBM Telco dataset.

    Learned imputations remain inside the sklearn pipeline. The optional
    ``drop_total_charges`` switch makes the redundancy ablation reproducible.
    '''
    path = path or DATASET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f'Dataset not found at {path}. '
            f'Place the IBM Telco CSV in {path.parent}.'
        )

    if drop_total_charges is None:
        drop_total_charges = DROP_TOTAL_CHARGES

    raw_df = pd.read_csv(path)
    audit = validate_raw_schema(raw_df)
    print(f'[data_loader] Loaded {raw_df.shape[0]:,} rows x {raw_df.shape[1]} cols')
    duplicate_id_count = audit['duplicate_ids']
    duplicate_row_count = audit['duplicate_rows']
    if duplicate_id_count or duplicate_row_count:
        print(
            f'[data_loader] WARNING: duplicate IDs={duplicate_id_count}, '
            f'duplicate rows={duplicate_row_count}'
        )

    df = raw_df.copy()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df = df.drop(columns=[ID_COL])

    # The source contains blank strings for new customers with tenure=0.
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_total_charges = int(df['TotalCharges'].isna().sum())
    if missing_total_charges:
        print(
            f'[data_loader] TotalCharges missing values: '
            f'{missing_total_charges} (handled by the pipeline)'
        )

    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
    )
    if df[TARGET_COL].isna().any():
        raise ValueError('Target column contains unmapped values')

    if drop_total_charges:
        df = df.drop(columns=['TotalCharges'])
        print('[data_loader] Primary policy: TotalCharges dropped')
    else:
        print('[data_loader] Ablation policy: TotalCharges retained')

    churn_rate = float(df[TARGET_COL].mean())
    print(
        f'[data_loader] Churn rate: {churn_rate:.2%} '
        f'({int(df[TARGET_COL].sum()):,} / {len(df):,})'
    )
    print(f'[data_loader] Features: {df.shape[1] - 1}')
    return df


def get_feature_target(df):
    '''Split a cleaned dataframe into features and target.'''
    return df.drop(columns=[TARGET_COL]), df[TARGET_COL]


def detect_feature_types(X):
    '''Return numeric and categorical columns in deterministic order.'''
    numeric = [column for column in NUMERIC_FEATURES if column in X.columns]
    categorical = [column for column in X.columns if column not in numeric]
    print(f'[data_loader] Numeric features ({len(numeric)}): {numeric}')
    print(f'[data_loader] Categorical features ({len(categorical)}): {categorical}')
    return numeric, categorical
