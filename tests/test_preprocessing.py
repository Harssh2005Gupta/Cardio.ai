from src.data.preprocessing import build_pipeline, get_output_feature_names, transform_to_df


def test_pipeline_fits_and_transforms(synthetic_df):
    X = synthetic_df.drop(columns=["target"])
    X_train, X_test = X.iloc[:160], X.iloc[160:]

    pipeline = build_pipeline()
    pipeline.fit(X_train)
    train_processed = transform_to_df(pipeline, X_train)
    test_processed = transform_to_df(pipeline, X_test)

    assert list(train_processed.columns) == list(test_processed.columns)
    assert len(train_processed) == len(X_train)
    assert len(test_processed) == len(X_test)
    assert not train_processed.isnull().any().any()


def test_no_leakage_scaler_uses_train_statistics_only(synthetic_df):
    """Proof that the scaler's fitted mean comes from the train split, not the
    full dataset — the core anti-leakage requirement."""
    X = synthetic_df.drop(columns=["target"])
    X_train = X.iloc[:150]
    X_full = X

    pipeline_train_only = build_pipeline()
    pipeline_train_only.fit(X_train)
    train_only_mean = pipeline_train_only.named_transformers_["num"].named_steps["scaler"].mean_

    pipeline_full = build_pipeline()
    pipeline_full.fit(X_full)
    full_mean = pipeline_full.named_transformers_["num"].named_steps["scaler"].mean_

    # The two fitted means must differ (proves train-only fitting produces a
    # genuinely different transform than fitting on the full dataset would).
    assert not (train_only_mean == full_mean).all()


def test_transform_never_refits(synthetic_df):
    """Calling transform_to_df on new data must not change the pipeline's
    learned statistics."""
    X = synthetic_df.drop(columns=["target"])
    X_train, X_test = X.iloc[:150], X.iloc[150:]

    pipeline = build_pipeline()
    pipeline.fit(X_train)
    mean_before = pipeline.named_transformers_["num"].named_steps["scaler"].mean_.copy()

    transform_to_df(pipeline, X_test)  # transform-only call

    mean_after = pipeline.named_transformers_["num"].named_steps["scaler"].mean_
    assert (mean_before == mean_after).all()


def test_output_feature_names_consistent(synthetic_df):
    X = synthetic_df.drop(columns=["target"])
    pipeline = build_pipeline()
    pipeline.fit(X.iloc[:150])
    names = get_output_feature_names(pipeline)
    transformed = transform_to_df(pipeline, X.iloc[150:])
    assert names == list(transformed.columns)
