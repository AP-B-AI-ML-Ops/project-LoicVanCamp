import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from train.scripts.preprocess_data import preprocess


def test_preprocess_transforms_categorical():
    df = pd.DataFrame({
        "gender": ["male", "female"],
        "race/ethnicity": ["group A", "group B"],
        "parental level of education": ["bachelor's", "master's"],
        "lunch": ["standard", "free/reduced"],
        "test preparation course": ["none", "completed"]
    })
    dv = DictVectorizer()
    X, dv_fitted = preprocess(df, dv, fit_dv=True)
    assert X.shape[0] == 2
    assert len(dv_fitted.feature_names_) > 0
