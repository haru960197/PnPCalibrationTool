"""
event_loader.py
---------------
イベントカメラデータの効率的な読み込みモジュール。

【ヒートマップ生成 (アノテーション用)】
  build_event_heatmap()   : events.csv を先頭行から順次読み込み、
                            指定した絶対時刻範囲のイベントを空間積算してヒートマップを返す。
  normalize_to_uint8()    : log1p 変換 → パーセンタイルクリップ → 0-255 線形スケールで画像化。

【単一フレーム生成 (verify_reprojection 用)】
  load_events_in_range()  : timestamp カラムのバイナリサーチで対象範囲を部分読み込み。
  build_event_frame()     : 極性で色分けした BGR フレームを生成。
"""

import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ヘッダー検出ユーティリティ
# ---------------------------------------------------------------------------

def _detect_header_rows(filepath: Path) -> int:
    """
    events.csv の先頭に存在するヘッダー/コメント行数を検出する。

    最初の列が数値にパースできない行をスキップ対象とする。
    "%geometry:320,320" のようなメタデータ行にも対応する。

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
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                skip += 1
                continue
            first_field = stripped.split(",")[0].strip()
            try:
                float(first_field)
                break
            except ValueError:
                skip += 1
    return skip


# ---------------------------------------------------------------------------
# ヒートマップ生成（アノテーション用）
# ---------------------------------------------------------------------------

def build_event_heatmap(
    filepath: str,
    start_delay_us: float,
    accumulation_time_us: float,
    width: int = 320,
    height: int = 320,
) -> np.ndarray:
    """
    events.csv の最初のイベント時刻から start_delay_us [µs] 遅らせた時刻を起点に、
    accumulation_time_us [µs] 分のイベントを空間的に積算してヒートマップを返す。

    極性は無視してイベント発生回数のみカウントする。

    アルゴリズム:
        1. CSV を 1 行ずつストリーミング読み込みし、メモリ効率を確保する。
        2. 最初のイベントのタイムスタンプを記録し、
           start_time_us = first_event_time_us + start_delay_us を計算する。
        3. [start_time_us, start_time_us + accumulation_time_us] の範囲のイベントを
           heatmap[y, x] += 1 でカウントする。

    Parameters
    ----------
    filepath : str
        events.csv のファイルパス
    start_delay_us : float
        最初のイベント時刻からの遅延 [µs]（キャリブレーション期間のスキップに使用）
    accumulation_time_us : float
        積算する時間幅 [µs]
    width : int
        センサ幅 [pixels]（デフォルト: 320）
    height : int
        センサ高さ [pixels]（デフォルト: 320）

    Returns
    -------
    np.ndarray, shape (height, width), dtype=float64
        各ピクセルのイベント発生回数を格納した 2D 配列。
        該当イベントがゼロの場合はゼロ行列を返す。
    """
    heatmap = np.zeros((height, width), dtype=np.float64)
    first_event_time_us: float | None = None
    start_time_us: float | None = None
    n_loaded = 0

    with open(filepath, "r", encoding="utf-8") as f:
        # ---- ヘッダー/メタデータ行のスキップ ----
        # "%geometry:320,320" のような行や列名行をスキップする。
        # define_led_region.py と同じロジックを採用。
        first_line = f.readline().strip()

        if first_line.startswith("%") or (first_line and not first_line[0].isdigit()):
            # geometry 情報があればセンササイズを上書き
            if "geometry" in first_line:
                try:
                    geo_part = first_line.split(":")[-1]
                    w_str, h_str = geo_part.split(",")
                    width = int(w_str.strip())
                    height = int(h_str.strip())
                    heatmap = np.zeros((height, width), dtype=np.float64)
                    print(f"[heatmap] センササイズ検出: {width}x{height}")
                except ValueError:
                    pass

            # 2 行目もヘッダかどうか確認
            second_line = f.readline().strip()
            if second_line and not second_line[0].isdigit():
                # カラム名行（例: "x,y,polarity,timestamp"）→ スキップしてそのまま続行
                pass
            else:
                # 2 行目がデータ行 → 先頭に巻き戻して再読み込み
                f.seek(0)
                f.readline()  # 1 行目（メタデータ）を再スキップ
        else:
            # 1 行目がデータ行（ヘッダなし）→ 先頭に戻す
            f.seek(0)

        # ---- ストリーミング積算 ----
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 4:
                continue
            try:
                x = int(row[0].strip())
                y = int(row[1].strip())
                # polarity (row[2]) は無視
                timestamp_us = float(row[3].strip())
            except (ValueError, IndexError):
                continue  # パース失敗行はスキップ（残存ヘッダ等）

            # 最初のイベント時刻を記録
            if first_event_time_us is None:
                first_event_time_us = timestamp_us
                start_time_us = first_event_time_us + start_delay_us
                print(
                    f"[heatmap] 最初のイベント: {first_event_time_us:.0f} µs  "
                    f"積算開始: {start_time_us:.0f} µs  "
                    f"(遅延 {start_delay_us:.0f} µs)"
                )

            # 積算開始前はスキップ
            if timestamp_us < start_time_us:
                continue

            # 積算終了時刻を超えたら打ち切り
            if timestamp_us - start_time_us > accumulation_time_us:
                break

            # 範囲内ピクセルをカウント
            if 0 <= x < width and 0 <= y < height:
                heatmap[y, x] += 1.0
                n_loaded += 1

    if start_time_us is not None:
        print(f"[heatmap] 積算完了: {n_loaded} イベント, 終了時刻 {timestamp_us:.0f} µs")
    else:
        print("[heatmap] 警告: イベントが一件も読み込まれませんでした。")

    return heatmap


def normalize_to_uint8(heatmap: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    """
    ヒートマップを対数変換・パーセンタイルクリップして 0-255 の uint8 グレースケール画像に変換する。

    処理手順:
        1. np.log1p によりホットピクセルの極端な値を圧縮する。
        2. clip_percentile パーセンタイル値を上限として np.clip でクリッピングする。
           最大値ではなくパーセンタイルを使うことで、ホットピクセル起因の正規化潰れを防ぐ。
        3. [0, clip_value] の範囲を [0, 255] にリニアスケールして uint8 に変換する。

    Parameters
    ----------
    heatmap : np.ndarray, shape (H, W), dtype=float64
        build_event_heatmap() が返すイベントカウント配列
    clip_percentile : float
        クリッピングに使うパーセンタイル値（例: 99.5）。
        これより高い値はすべて飽和扱いになる。

    Returns
    -------
    np.ndarray, shape (H, W), dtype=uint8
        正規化済みのグレースケール画像（0 = 無イベント、255 = 高頻度）
    """
    # Step 1: 対数変換（log1p で 0 を安全に扱う）
    log_heatmap = np.log1p(heatmap)

    if log_heatmap.max() == 0.0:
        return np.zeros_like(heatmap, dtype=np.uint8)

    # Step 2: パーセンタイルでクリッピング値を決定
    clip_value = float(np.percentile(log_heatmap, clip_percentile))

    # クリップ値が極小の場合は最大値にフォールバック
    if clip_value <= 1e-8:
        clip_value = float(log_heatmap.max())
        if clip_value <= 1e-8:
            return np.zeros_like(heatmap, dtype=np.uint8)

    print(
        f"[normalize] log1p max={log_heatmap.max():.4f}, "
        f"clip@{clip_percentile}%={clip_value:.4f}"
    )

    # Step 3: クリップ → 線形スケール → uint8
    clipped = np.clip(log_heatmap, 0.0, clip_value)
    normalized = (clipped / clip_value * 255.0).astype(np.uint8)
    return normalized


def heatmap_to_bgr(gray: np.ndarray, colormap: int = -1) -> np.ndarray:
    """
    グレースケールのヒートマップ画像を BGR カラー画像に変換する。

    colormap に OpenCV のカラーマップ定数（例: cv2.COLORMAP_JET）を指定すると
    疑似カラー表示になる。-1 を指定するとグレースケール（3ch）のまま返す。

    Parameters
    ----------
    gray : np.ndarray, shape (H, W), dtype=uint8
        normalize_to_uint8() が返すグレースケール画像
    colormap : int
        OpenCV カラーマップ定数（デフォルト: -1 = グレースケール）

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype=uint8
        BGR カラー画像
    """
    if colormap >= 0:
        import cv2
        return cv2.applyColorMap(gray, colormap)
    return np.stack([gray, gray, gray], axis=-1)


# ---------------------------------------------------------------------------
# 単一イベントフレーム生成（verify_reprojection 用）
# ---------------------------------------------------------------------------

def load_events_in_range(
    filepath: str,
    t_start_us: float,
    t_end_us: float,
) -> np.ndarray:
    """
    指定したタイムスタンプ範囲 [t_start_us, t_end_us] のイベントを
    バイナリサーチで効率的に読み込んで返す。

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

    正のイベント（polarity=1）を白（255）、負のイベント（polarity=0 or -1）を
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
        return np.stack([frame, frame, frame], axis=-1)

    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    pols = events[:, 2].astype(np.int32)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs, ys, pols = xs[valid], ys[valid], pols[valid]

    frame[ys, xs] = np.where(pols > 0, 255, 128)

    return np.stack([frame, frame, frame], axis=-1)
