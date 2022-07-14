import pandas as pd
from ipdb import set_trace as ist


def frame2sec(frame: int, fps: float = 29.97):
    """フレーム番号からmsを計算
    Args:
        frame (int): フレーム番号
        fps (float): frame per second
    Returns:
        秒数
    """
    return frame * 1 / fps


def sec2frame(sec: float, fps: float = 29.97):
    """msからフレーム番号を計算
    Args:
        sec (float): 秒数
        fps (float): frame per second
    Returns:
        フレーム番号
    """
    return max(0, int(sec * fps))


def groupby_rally(
    input_df: pd.DataFrame, col: str = "start", threshold: int = 50, fps: float = 29.97
):
    """dfのフレーム間差分から、１つのラリーを検出する
    Args:
        input_df (pd.DataFrame): [start,end,labels]
        col (str): アノテーションの開始秒数
        threshold (int): ラリーの閾値
        fps (float): 変換するときのfps
    Returns:
        pd.DataFrame: [start,end,labels,start_frame,start_frame_diff,group]
    """
    output_df = input_df.reset_index(drop=True).copy()
    output_df = _df_sec2frame(output_df, col=col, fps=fps)
    output_df = _df_diff(output_df, col=f"{col}_frame")

    group_id = 0
    output_df = output_df.reset_index(drop=True)
    output_df.loc[0, "group"] = 0
    for i in range(1, len(output_df)):
        if output_df.loc[i, f"{col}_frame_diff"] >= threshold:
            group_id += 1
        output_df.loc[i, "group"] = group_id

    return output_df


def _df_sec2frame(input_df: pd.DataFrame, col="start", fps=29.97):
    """
    秒をフレームに変更する
    """
    output_df = input_df.copy()
    output_df[f"{col}_frame"] = output_df[col].apply(lambda x: sec2frame(x, fps=fps))
    return output_df


def _df_diff(input_df: pd.DataFrame, col="start_frame"):
    """フレーム間差分を求める"""
    output_df = input_df.copy()
    output_df[f"{col}_diff"] = output_df[col].diff()
    return output_df


if __name__ == "__main__":
    print(frame2sec(3))
    print(frame2sec(30))
    print(frame2sec(300))
    print(frame2sec(2 * 60 + 4))

    print(sec2frame(1.984317))
    print(sec2frame(2.515939))

    df = pd.DataFrame(
        {
            "start": {
                0: 77.76126532585296,
                1: 78.39545904293517,
                2: 78.9787198568589,
                3: 79.56105305174262,
                4: 91.54197714717344,
                5: 92.19189633983132,
                6: 92.7334956670462,
                7: 93.1149261066875,
                8: 93.59819935251005,
                9: 109.02961402946349,
                10: 109.7711884928808,
                11: 110.38777849617159,
                12: 110.68774120047522,
                13: 111.35432498781664,
                14: 125.55718961637938,
                15: 126.22562864180077,
                16: 126.72093652411594,
                17: 127.19217513344582,
                18: 141.31818571084352,
                19: 141.97367061774142,
                20: 142.68191589179162,
                21: 143.05408641505116,
                22: 157.13937939879833,
                23: 157.77263399677258,
                24: 158.2642395399369,
                25: 159.1085681545572,
                26: 159.72515815784797,
                27: 160.05845005151866,
                28: 160.55654892796298,
                29: 184.6868820297217,
            },
            "end": {
                0: 77.86125302452525,
                1: 78.4704497190111,
                2: 79.04537823559305,
                3: 79.60271453845147,
                4: 91.58363863388227,
                5: 92.2502224212237,
                6: 92.80015404578036,
                7: 93.16491989073812,
                8: 93.6565254339024,
                9: 109.0712755161723,
                10: 109.85451146629845,
                11: 110.4627691722475,
                12: 110.7460672818676,
                13: 111.412651069209,
                14: 125.61551569777174,
                15: 126.31728391256023,
                16: 126.79592720019184,
                17: 127.2754981068635,
                18: 141.40984098160294,
                19: 142.06532588850084,
                20: 142.78190345989285,
                21: 143.1290770911271,
                22: 157.23103466955772,
                23: 157.84762467284853,
                24: 158.33089791867104,
                25: 159.16689423594954,
                26: 159.8084811312656,
                27: 160.15843761961992,
                28: 160.63153960403892,
                29: 184.7868695978229,
            },
            "labels": {
                0: "OA",
                1: "XB",
                2: "XA",
                3: "OB",
                4: "OA",
                5: "XB",
                6: "OA",
                7: "XB",
                8: "OA",
                9: "OB",
                10: "OA",
                11: "OB",
                12: "XA",
                13: "OB",
                14: "OB",
                15: "OA",
                16: "XB",
                17: "XA",
                18: "OA",
                19: "XB",
                20: "OA",
                21: "XB",
                22: "OA",
                23: "XB",
                24: "XA",
                25: "XB",
                26: "OA",
                27: "XB",
                28: "OA",
                29: "OB",
            },
        }
    )

    groupby_rally(df, threshold=50, fps=25)

    pass
