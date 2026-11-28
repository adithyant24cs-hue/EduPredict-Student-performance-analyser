import sys
import pandas as pd
sys.path.insert(0, "app")
from predictor import load_model, get_required_features

pipe = load_model()
df = pd.read_csv(sys.argv[1])
target_col = sys.argv[2]

X = df[get_required_features(pipe)]
df["prob_class_1"] = pipe.predict_proba(X)[:, 1]
print(df.groupby(target_col)["prob_class_1"].agg(["mean", "count"]).round(3))