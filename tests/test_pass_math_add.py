import pandas as pd
from deploy_batch.batch import read_dataframe


def test_read_dataframe_adds_pass_math(tmp_path):
    df = pd.DataFrame({"math score": [40, 60]})
    file = tmp_path / "students.csv"
    df.to_csv(file, index=False)
    result = read_dataframe(str(file))
    assert "pass_math" in result.columns
    assert list(result["pass_math"]) == [0, 1]
