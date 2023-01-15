import pandas as pd


def util_encoder(df: pd.DataFrame) -> dict[pd.DataFrame, list]:
    multi_class_features = []
    for col in df.columns:
        col_classes = df[col].unique()
        print(col_classes)
        if len(col_classes) > 1:
            multi_class_features.append(col)
    results_df = pd.get_dummies(df[multi_class_features])
    results_df['transferred'] = results_df['transferred'].astype('int')
    return {'dataframe': results_df, 'features': multi_class_features}
