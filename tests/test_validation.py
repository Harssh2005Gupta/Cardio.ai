from src.data.validation import validate_dataset, validate_patient_record


def test_valid_dataset_passes(synthetic_df):
    result = validate_dataset(synthetic_df)
    assert result.is_valid, result.errors


def test_missing_columns_rejected(synthetic_df):
    bad = synthetic_df.drop(columns=["age"])
    result = validate_dataset(bad)
    assert not result.is_valid
    assert any("age" in e for e in result.errors)


def test_empty_dataset_rejected(synthetic_df):
    result = validate_dataset(synthetic_df.iloc[0:0])
    assert not result.is_valid


def test_invalid_categorical_value_rejected(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "sex"] = 7  # not in {0, 1}
    result = validate_dataset(bad)
    assert not result.is_valid


def test_non_binary_target_rejected(synthetic_df):
    bad = synthetic_df.copy()
    bad.loc[0, "target"] = 2
    result = validate_dataset(bad)
    assert not result.is_valid


def test_valid_patient_record_passes():
    record = {
        "age": 58, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0,
        "restecg": 1, "thalach": 145, "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
    }
    result = validate_patient_record(record)
    assert result.is_valid, result.errors


def test_patient_record_missing_field_rejected():
    record = {"age": 58, "sex": 1}
    result = validate_patient_record(record)
    assert not result.is_valid


def test_patient_record_out_of_range_rejected():
    record = {
        "age": 999, "sex": 1, "cp": 2, "trestbps": 140, "chol": 289, "fbs": 0,
        "restecg": 1, "thalach": 145, "exang": 1, "oldpeak": 2.3, "slope": 1, "ca": 1, "thal": 3,
    }
    result = validate_patient_record(record)
    assert not result.is_valid
