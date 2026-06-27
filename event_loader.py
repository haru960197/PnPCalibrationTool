"""
event_loader.py
---------------
イベントカメラデータの効率的な読み込みモジュール。

巨大な events.csv を扱うため、以下の戦略を採用する:
  1. まず全体を timestamp カラムのみ pandas でスキャンし、
     バイナリサーチで対象範囲の行番号を特定する。
  2. 対象範囲のみを skiprows/nrows で部分読み込みする。
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def _detect_header_rows(filepath: Path) -> int:
    """
    events.csv の先頭に存在するヘッダー/コメント行数を検出する。

    最初の数行を読み、最初の列が数値にパースできない行をスキップ対象とする。

    Parameters
    ----------
    filepath : Path
        events.csv のファイルパス

    Returns
    -------
    int
        スキップすべき行数（データヘッダー行含む）
    """
    skip = 0
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                skip += 1
                continue
            # 最初のフィールドが数値かどうかでデータ行を判定
            first_field = stripped.split(",")[0]
            try:
                float(first_field)
                break
            except ValueError:
                skip += 1
    return skip


def load_events_in_range(
    filepath: str,
    t_start_us: float,
    t_end_us: float,
) -> np.ndarray:
    """
    指定したタイムスタンプ範囲 [t_start_us, t_end_us] のイベントを
    効率的に読み込んで返す。

    events.csv のカラム順は x, y, polarity, timestamp を想定。
    timestamp カラムの単位はマイクロ秒（µs）。

    Parameters
    ----------
    filepath : str
        events.csv のファイルパス
    t_start_us : float
        抽出開始タイムスタンプ [µs]
    t_end_us : float
        抽出終了タイムスタンプ [µs]

    Returns
    -------
    np.ndarray, shape (N, 4)
        各行が [x, y, polarity, timestamp] の NumPy 配列。
        該当イベントが存在しない場合は shape (0, 4) を返す。
    """
    fpath = Path(filepath)
    skip_rows = _detect_header_rows(fpath)

    # ---- Step 1: timestamp カラムのみを高速スキャン ----
    # usecols で必要列のみ読み込み、メモリ節約
    try:
        ts_df = pd.read_csv(
            fpath,
            skiprows=skip_rows,
            header=None,
            names=["x", "y", "polarity", "timestamp"],
            usecols=["timestamp"],
            dtype={"timestamp": np.float64},
            engine="c",
        )
    except Exception as e:
        raise RuntimeError(f"events.csv の読み込みに失敗しました: {e}") from e

    timestamps = ts_df["timestamp"].values

    if len(timestamps) == 0:
        return np.empty((0, 4), dtype=np.float64)

    # ---- Step 2: バイナリサーチで行インデックスを特定 ----
    idx_start = int(np.searchsorted(timestamps, t_start_us, side="left"))
    idx_end = int(np.searchsorted(timestamps, t_end_us, side="right"))

    if idx_start >= idx_end:
        return np.empty((0, 4), dtype=np.float64)

    # ---- Step 3: 対象行のみ部分読み込み ----
    # skiprows: ヘッダー分 + データ先頭からのオフセット
    actual_skip = skip_rows + idx_start
    nrows = idx_end - idx_start

    chunk = pd.read_csv(
        fpath,
        skiprows=actual_skip,
        nrows=nrows,
        header=None,
        names=["x", "y", "polarity", "timestamp"],
        dtype={
            "x": np.int32,
            "y": np.int32,
            "polarity": np.int8,
            "timestamp": np.float64,
        },
        engine="c",
    )

    return chunk[["x", "y", "polarity", "timestamp"]].values


def build_event_frame(
    events: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    イベント配列からグレースケールの可視化フレーム（BGR画像）を生成する。

    正のイベント（polarity=1）を白、負のイベント（polarity=0 or -1）を
    グレー（128）で描画し、背景を黒とする。

    Parameters
    ----------
    events : np.ndarray, shape (N, 4)
        各行が [x, y, polarity, timestamp] のイベント配列
    width : int
        出力フレームの幅 [pixels]
    height : int
        出力フレームの高さ [pixels]

    Returns
    -------
    np.ndarray, shape (height, width, 3)
        BGR カラー画像（uint8）
    """
    frame = np.zeros((height, width), dtype=np.uint8)

    if len(events) == 0:
        # イベントがない場合は黒画像を返す
        return cv2_gray_to_bgr(frame)

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    pols = events[:, 2].astype(np.int32)

    # 範囲外の座標をクリップ
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, pols = xs[valid], ys[valid], pols[valid]

    # 負イベント: 128, 正イベント: 255
    frame[ys, xs] = np.where(pols > 0, 255, 128)

    # BGR に変換（OpenCV 用）
    return cv2_gray_to_bgr(frame)


def cv2_gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """
    グレースケール画像を BGR 3チャンネル画像へ変換する。

    Parameters
    ----------
    gray : np.ndarray, shape (H, W)
        グレースケール画像

    Returns
    -------
    np.ndarray, shape (H, W, 3)
        BGR 画像
    """
    return np.stack([gray, gray, gray], axis=-1)
