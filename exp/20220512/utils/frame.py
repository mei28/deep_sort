def frame2sec(frame: int, fps: float = 29.97):
    """フレーム番号からmsを計算
    Args:
        frame (int): フレーム番号
        fps (float): frame per second
    Returns:
        秒数
    """
    return fps * frame


def sec2frame(sec: float, fps: float = 29.97):
    """msからフレーム番号を計算
    Args:
        sec (float): 秒数
        fps (float): frame per second
    Returns:
        フレーム番号
    """
    return max(0, int(sec * fps))


if __name__ == "__main__":
    print(frame2sec(3))
    print(frame2sec(30))
    print(frame2sec(300))
    print(frame2sec(2 * 60 + 4))

    print(sec2frame(1.984317))
    print(sec2frame(2.515939))
    pass
