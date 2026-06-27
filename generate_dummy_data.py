"""
generate_dummy_data.py
----------------------
動作確認用のダミー入力データを生成するスクリプト。

実際のデータが手元にない場合、このスクリプトを実行すると
input/ ディレクトリに必要なファイルが作成され、
main_pnp_solver.py および verify_reprojection.py の動作テストができる。

使い方:
  python generate_dummy_data.py
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_sync_log(n_frames: int = 30, fps: float = 30.0) -> None:
    """
    sync_log.csv を生成する。

    Parameters
    ----------
    n_frames : int
        生成するフレーム数
    fps : float
        フレームレート [fps]
    """
    records = []
    for i in range(n_frames):
        ts_ms = i * (1000.0 / fps)
        led = 1 if (i // 5) % 2 == 0 else 0
        records.append({"frame_index": i, "timestamp_ms": round(ts_ms, 3), "led_status": led})
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "sync_log.csv", index=False)
    print(f"[生成] {OUTPUT_DIR / 'sync_log.csv'} ({n_frames} フレーム)")


def generate_sync_params() -> dict:
    """
    sync_params.json を生成する。

    RGBカメラとイベントカメラの時間同期パラメータ。
    t_rgb [ms] = A * t_event [µs] + B
    """
    params = {"A": 0.001, "B": 0.0}  # t_event [µs] → t_rgb [ms] に変換するパラメータ
    with open(OUTPUT_DIR / "sync_params.json", "w") as f:
        json.dump(params, f, indent=2)
    print(f"[生成] {OUTPUT_DIR / 'sync_params.json'}")
    return params


def generate_events(total_duration_us: float = 15_000_000, events_per_ms: int = 20) -> None:
    """
    events.csv を生成する（ランダムなイベントデータ）。

    config.json の start_delay_us=7,000,000 µs + accumulation_time_us=5,000,000 µs に
    対応するため、0 ～ total_duration_us の範囲でイベントを生成する。
    ヒートマップ積算対象となる 7,000,000 ～ 12,000,000 µs の範囲に
    十分なイベントが含まれるようにする。

    Parameters
    ----------
    total_duration_us : float
        生成するイベント全体の時間幅 [µs]（デフォルト: 15 秒）
    events_per_ms : int
        1ms あたりの平均イベント数
    """
    rng = random.Random(42)
    all_rows = []

    total_ms = int(total_duration_us / 1000)
    for ms_idx in range(total_ms):
        t_center_us = ms_idx * 1000.0
        for _ in range(events_per_ms):
            x = rng.randint(0, 319)
            y = rng.randint(0, 319)
            pol = rng.choice([0, 1])
            t = t_center_us + rng.uniform(0, 1000)
            all_rows.append((x, y, pol, t))

    # timestamp でソート
    all_rows.sort(key=lambda r: r[3])

    df = pd.DataFrame(all_rows, columns=["x", "y", "polarity", "timestamp"])
    df.to_csv(OUTPUT_DIR / "events.csv", index=False)
    print(f"[生成] {OUTPUT_DIR / 'events.csv'} ({len(all_rows)} イベント, {total_duration_us/1e6:.1f} 秒分)")


def generate_landmark(n_frames: int = 30, n_landmarks: int = 478) -> None:
    """
    landmark.csv を生成する（正規化座標のダミーデータ）。

    Parameters
    ----------
    n_frames : int
        フレーム数
    n_landmarks : int
        ランドマーク数（MediaPipe 顔メッシュは 478 点）
    """
    rng = np.random.default_rng(42)
    rows = []
    # 顔の大まかな形を模倣するためランドマークごとに固定オフセットを持たせる
    base_positions = rng.uniform(-0.2, 0.2, size=(n_landmarks, 3))
    base_positions[:, 0] += 0.0    # x: 中央付近
    base_positions[:, 1] += 0.0    # y: 中央付近
    base_positions[:, 2] -= 0.05   # z: 少し奥

    for frame_idx in range(n_frames):
        # フレームごとに微小な摂動
        noise = rng.normal(0, 0.002, size=(n_landmarks, 3))
        positions = base_positions + noise

        for lm_idx in range(n_landmarks):
            rows.append({
                "frame_index": frame_idx,
                "face_index": 0,
                "landmark_index": lm_idx,
                "x_norm": round(float(positions[lm_idx, 0]), 6),
                "y_norm": round(float(positions[lm_idx, 1]), 6),
                "z_norm": round(float(positions[lm_idx, 2]), 6),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "landmark.csv", index=False)
    print(f"[生成] {OUTPUT_DIR / 'landmark.csv'} ({len(rows)} 行)")


def generate_calibration() -> None:
    """
    calibration.json を生成する（320x320 イベントカメラ用のダミー内部パラメータ）。
    """
    fx = fy = 200.0
    cx, cy = 160.0, 160.0
    intrinsics = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    distortion = [0.0, 0.0, 0.0, 0.0, 0.0]

    cal = {"intrinsics": intrinsics, "distortion": distortion}
    with open(OUTPUT_DIR / "calibration.json", "w") as f:
        json.dump(cal, f, indent=2)
    print(f"[生成] {OUTPUT_DIR / 'calibration.json'}")


if __name__ == "__main__":
    print("=== ダミーデータ生成 ===")
    generate_sync_log()
    generate_sync_params()
    print("[注意] events.csv は 15 秒分（約 300,000 イベント）を生成します。少々お待ちください...")
    generate_events()
    generate_landmark()
    generate_calibration()
    print("\n完了! input/ ディレクトリにファイルを生成しました。")
    print("次のコマンドでメインツールを起動できます:")
    print("  python main_pnp_solver.py --config config.json")
