"""Unit test for verifying categorical feature transformation using preprocess."""

# pylint: disable=invalid-name

import pandas as pd
from sklearn.feature_extraction import DictVectorizer

from train.scripts.preprocess_data import preprocess


def test_preprocess_transforms_categorical():
    """Test that preprocess transforms categorical features as expected.

    This test checks that the DictVectorizer transforms the categorical columns
    into the correct number of features.
    """
    df = pd.DataFrame(
        {
            "gender": ["male", "female"],
            "race/ethnicity": ["group A", "group B"],
            "parental level of education": ["bachelor's", "master's"],
            "lunch": ["standard", "free/reduced"],
            "test preparation course": ["none", "completed"],
        }
    )
    dv = DictVectorizer()
    X, dv_fitted = preprocess(df, dv, fit_dv=True)
    assert X.shape[0] == 2
    # Compatibiliteit met verschillende scikit-learn versies
    if hasattr(dv_fitted, "feature_names_out_"):
        n_features = len(dv_fitted.feature_names_out_)
    else:
        n_features = len(dv_fitted.feature_names_)
    assert X.shape[1] == n_features
