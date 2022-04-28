import pandas as pd
import json


def load_data(path: str):
    df = pd.read_csv(path)
    js = json.loads(df["tricks"][0])
    return js


def list2df(data):
    output_df = pd.DataFrame()
    for d in data:
        _df = pd.DataFrame.from_dict(d)
        output_df = pd.concat([output_df, _df], axis=0).reset_index(drop=True)
    return output_df


if __name__ == "__main__":
    pass
