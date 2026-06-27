"""
verify_reprojection.py
----------------------
PnP 計算結果の検証ツール。

main_pnp_solver.py が出力した transform_matrix.json の rvec / tvec を読み込み、
landmark.csv に含まれる全（または主要な）3D 顔メッシュ点をイベントフレームへ
再投影（リプロジェクション）して目視確認する。

使い方:
  python verify_reprojection.py [--config CONFIG] [--frame FRAME_INDEX]

キーボード操作:
  n / →  : 次のフレームへ（同じ rvec/tvec で再投影）
  p / ←  : 前のフレームへ
  q / Esc: 終了
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from event_loader import build_event_frame, load_events_in_range
from main_pnp_solver import (
    DISPLAY_SCALE,
    FONT,
    TEXT_BG_COLOR,
    TEXT_COLOR,
    WINDOW_NAME,
    get_event_frame_for_rgb_frame,
    load_calibration,
    load_config,
    load_landmark_csv,
    load_sync_log,
    load_sync_params,
)

VERIFY_WINDOW_NAME = "Reprojection Verification"
PROJ_COLOR = (0, 0, 255)     # 再投影点の色（赤）[BGR]
ANNO_COLOR = (0, 255, 0)     # アノテーション済み点の色（緑）[BGR]
PROJ_RADIUS = 2              # 再投影点の円半径 [pixels]
ANNO_RADIUS = 5              # アノテーション点の円半径 [pixels]


# ---------------------------------------------------------------------------
# データ読み込みユーティリティ
# ---------------------------------------------------------------------------

def load_transform(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    transform_matrix.json を読み込み、rvec, tvec, frame_index を返す。

    Parameters
    ----------
    path : str
        transform_matrix.json のパス

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (rvec [3x1], tvec [3x1], frame_index)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(data["tvec"], dtype=np.float64).reshape(3, 1)
    frame_index = int(data["frame_index"])

    print(f"[変換行列] frame_index={frame_index}")
    print(f"  rvec: {rvec.ravel()}")
    print(f"  tvec: {tvec.ravel()}")

    return rvec, tvec, frame_index


def load_annotated_points(path: str) -> Optional[Dict]:
    """
    annotated_points.json を読み込む（存在しない場合は None）。

    Parameters
    ----------
    path : str
        annotated_points.json のパス

    Returns
    -------
    Optional[Dict]
        アノテーション済み点データ
    """
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_all_3d_points_for_frame(
    landmark_df: pd.DataFrame,
    frame_index: int,
) -> Tuple[np.ndarray, List[int]]:
    """
    指定フレームの全ランドマークの 3D 座標を取得する。

    Parameters
    ----------
    landmark_df : pd.DataFrame
        landmark.csv の DataFrame
    frame_index : int
        対象フレームインデックス

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        - points_3d: shape (N, 3) の float64 配列
        - landmark_ids: ランドマーク ID のリスト（N 要素）
    """
    frame_df = landmark_df[landmark_df["frame_index"] == frame_index].copy()
    frame_df = frame_df.sort_values("landmark_index")

    if frame_df.empty:
        return np.empty((0, 3), dtype=np.float64), []

    points_3d = frame_df[["x_norm", "y_norm", "z_norm"]].values.astype(np.float64)
    landmark_ids = frame_df["landmark_index"].tolist()

    return points_3d, landmark_ids


# ---------------------------------------------------------------------------
# 再投影と描画
# ---------------------------------------------------------------------------

def project_and_draw(
    base_frame: np.ndarray,
    points_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    annotated_points: Optional[Dict] = None,
    frame_index: Optional[int] = None,
    source_frame_index: Optional[int] = None,
) -> np.ndarray:
    """
    3D 点をイベントフレームに再投影して描画した画像を返す。

    Parameters
    ----------
    base_frame : np.ndarray
        描画対象のベース BGR 画像
    points_3d : np.ndarray, shape (N, 3)
        再投影対象の 3D 点群
    rvec : np.ndarray, shape (3, 1)
        回転ベクトル
    tvec : np.ndarray, shape (3, 1)
        並進ベクトル
    camera_matrix : np.ndarray, shape (3, 3)
        イベントカメラのカメラ行列
    dist_coeffs : np.ndarray
        イベントカメラの歪み係数
    annotated_points : Optional[Dict]
        アノテーション済み点データ（annotated_points.json の内容）
    frame_index : Optional[int]
        現在表示中のフレームインデックス（情報表示用）
    source_frame_index : Optional[int]
        rvec/tvec を計算した元のフレームインデックス（情報表示用）

    Returns
    -------
    np.ndarray
        オーバーレイ済み BGR 画像（表示倍率適用後）
    """
    img = base_frame.copy()
    h, w = img.shape[:2]

    # --- 再投影 ---
    if len(points_3d) > 0:
        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        for pt in projected:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), PROJ_RADIUS, PROJ_COLOR, -1)

    # --- アノテーション済み点を緑で重ね描き ---
    if annotated_points is not None:
        for lm in annotated_points.get("landmarks", []):
            u, v = lm["point_2d"]
            x, y = int(round(u)), int(round(v))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), ANNO_RADIUS, ANNO_COLOR, 2)
            cv2.putText(
                img, lm["name"], (x + 6, y - 4),
                FONT, 0.32, ANNO_COLOR, 1, cv2.LINE_AA
            )

    # --- 情報テキスト ---
    n_proj = len(points_3d)
    info1 = (
        f"Frame: {frame_index}  "
        f"(PnP source: {source_frame_index})  "
        f"Projected: {n_proj} pts"
    )
    info2 = "Red: projected  Green: annotated  [n]次 [p]前 [q]終了"

    cv2.rectangle(img, (0, 0), (w, 38), TEXT_BG_COLOR, -1)
    cv2.putText(img, info1, (4, 14), FONT, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, info2, (4, 32), FONT, 0.36, (180, 180, 100), 1, cv2.LINE_AA)

    # 表示倍率適用
    disp_w = w * DISPLAY_SCALE
    disp_h = h * DISPLAY_SCALE
    img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    return img


# ---------------------------------------------------------------------------
# メイン検証クラス
# ---------------------------------------------------------------------------

class ReprojectionVerifier:
    """
    rvec / tvec を使って顔メッシュ全点をイベントフレームに再投影し確認するクラス。

    Attributes
    ----------
    config : Dict
        設定辞書
    rvec : np.ndarray
        回転ベクトル [3x1]
    tvec : np.ndarray
        並進ベクトル [3x1]
    source_frame_index : int
        PnP 計算に使用した元のフレームインデックス
    camera_matrix : np.ndarray
        イベントカメラの内部パラメータ行列 [3x3]
    dist_coeffs : np.ndarray
        イベントカメラの歪み係数 [5x1]
    """

    def __init__(self, config: Dict, config_dir: Path):
        """
        Parameters
        ----------
        config : Dict
            config.json を解析した辞書
        config_dir : Path
            config.json が置かれているディレクトリ
        """
        self.config = config
        self.config_dir = config_dir
        self._resolve = lambda p: str(config_dir / p)

        paths = config["paths"]

        print("[初期化] sync_log.csv を読み込み中...")
        self.sync_log = load_sync_log(self._resolve(paths["sync_log"]))

        print("[初期化] sync_params.json を読み込み中...")
        self.A, self.B = load_sync_params(self._resolve(paths["sync_params"]))

        print("[初期化] landmark.csv を読み込み中...")
        self.landmark_df = load_landmark_csv(self._resolve(paths["landmark"]))

        print("[初期化] calibration.json を読み込み中...")
        self.camera_matrix, self.dist_coeffs = load_calibration(
            self._resolve(paths["calibration"])
        )

        print("[初期化] transform_matrix.json を読み込み中...")
        self.rvec, self.tvec, self.source_frame_index = load_transform(
            self._resolve(paths["output_transform"])
        )

        self.annotated_points = load_annotated_points(
            self._resolve(paths["output_points"])
        )

        self.events_path = self._resolve(paths["events"])

        ef_cfg = config["event_frame"]
        self.width = int(ef_cfg["width"])
        self.height = int(ef_cfg["height"])
        self.integration_time_ms = float(ef_cfg["integration_time_ms"])

        self.frame_indices = sorted(self.sync_log.index.tolist())
        self.current_frame_pos = 0

        # イベントフレームキャッシュ
        self._frame_cache: Dict[int, np.ndarray] = {}

        print("[初期化] 完了")

    def _get_event_frame(self, frame_index: int) -> np.ndarray:
        """
        指定フレームのイベントフレームをキャッシュ付きで取得する。

        Parameters
        ----------
        frame_index : int
            RGB フレームインデックス

        Returns
        -------
        np.ndarray
            BGR イベントフレーム画像
        """
        if frame_index not in self._frame_cache:
            print(f"  [フレーム生成] frame_index={frame_index} ...")
            frame = get_event_frame_for_rgb_frame(
                frame_index=frame_index,
                sync_log=self.sync_log,
                A=self.A,
                B=self.B,
                events_path=self.events_path,
                width=self.width,
                height=self.height,
                integration_time_ms=self.integration_time_ms,
            )
            self._frame_cache[frame_index] = frame
        return self._frame_cache[frame_index].copy()

    def run(self, initial_frame_index: Optional[int] = None):
        """
        GUI メインループを起動する。

        Parameters
        ----------
        initial_frame_index : Optional[int]
            起動時に表示する RGB フレームインデックス。
            None の場合は PnP 計算に使用したフレームから開始。
        """
        if not self.frame_indices:
            print("[エラー] 利用可能なフレームがありません。")
            return

        # 初期フレーム位置
        start_idx = initial_frame_index if initial_frame_index is not None \
            else self.source_frame_index

        if start_idx in self.frame_indices:
            self.current_frame_pos = self.frame_indices.index(start_idx)
        else:
            self.current_frame_pos = 0

        cv2.namedWindow(VERIFY_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        print("\n=== Reprojection Verifier 操作ガイド ===")
        print("  n / →  : 次のフレームへ（同じ rvec/tvec で再投影）")
        print("  p / ←  : 前のフレームへ")
        print("  q / Esc: 終了")
        print("=========================================\n")

        need_redraw = True

        while True:
            frame_index = self.frame_indices[self.current_frame_pos]

            if need_redraw:
                # イベントフレーム取得
                base_frame = self._get_event_frame(frame_index)

                # 全 3D 点を取得
                points_3d, landmark_ids = get_all_3d_points_for_frame(
                    self.landmark_df, frame_index
                )
                print(f"  [再投影] frame_index={frame_index}: {len(points_3d)} 点")

                # 再投影・描画
                display = project_and_draw(
                    base_frame=base_frame,
                    points_3d=points_3d,
                    rvec=self.rvec,
                    tvec=self.tvec,
                    camera_matrix=self.camera_matrix,
                    dist_coeffs=self.dist_coeffs,
                    annotated_points=self.annotated_points,
                    frame_index=frame_index,
                    source_frame_index=self.source_frame_index,
                )

                cv2.imshow(VERIFY_WINDOW_NAME, display)
                need_redraw = False

            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                if cv2.getWindowProperty(VERIFY_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[終了] ウィンドウが閉じられました。")
                    break
                continue

            elif key in (ord("n"), 83):  # 次フレーム
                if self.current_frame_pos < len(self.frame_indices) - 1:
                    self.current_frame_pos += 1
                    need_redraw = True
                else:
                    print("[情報] 最後のフレームです。")

            elif key in (ord("p"), 81):  # 前フレーム
                if self.current_frame_pos > 0:
                    self.current_frame_pos -= 1
                    need_redraw = True
                else:
                    print("[情報] 最初のフレームです。")

            elif key in (ord("q"), 27):  # 終了
                print("[終了] Verifier を終了します。")
                break

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main():
    """メインエントリーポイント。コマンドライン引数を解析して検証 GUI を起動する。"""
    parser = argparse.ArgumentParser(
        description=(
            "Reprojection Verifier: PnP 変換行列を使って顔メッシュをイベントフレームに"
            "再投影して目視確認する。"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="config.json のパス (デフォルト: config.json)",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help=(
            "起動時に表示する RGB フレームインデックス "
            "(デフォルト: PnP 計算に使用したフレーム)"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[エラー] config.json が見つかりません: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    config_dir = config_path.parent

    verifier = ReprojectionVerifier(config, config_dir)
    verifier.run(initial_frame_index=args.frame)


if __name__ == "__main__":
    main()
