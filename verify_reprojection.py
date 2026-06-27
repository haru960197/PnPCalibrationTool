"""
verify_reprojection.py
----------------------
PnP 計算結果の検証ツール。

main_pnp_solver.py が出力した transform_matrix.json の rvec / tvec を読み込み、
landmark.csv に含まれる全（または主要な）3D 顔メッシュ点をヒートマップへ
再投影（リプロジェクション）して目視確認する。

【表示内容】
  - 赤い小円 : PnP で求めた変換行列で投影した全顔メッシュ点
  - 緑の大円 : アノテーション時にクリックした 7 点（annotated_points.json から）

使い方:
  python verify_reprojection.py [--config CONFIG]

キーボード操作:
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

from event_loader import (
    build_event_heatmap,
    heatmap_to_bgr,
    normalize_to_uint8,
)
from main_pnp_solver import (
    DISPLAY_SCALE,
    FONT,
    TEXT_BG_COLOR,
    TEXT_COLOR,
    get_all_3d_points_for_frame,
    load_calibration,
    load_config,
    load_landmark_csv,
    build_annotation_heatmap,
)

VERIFY_WINDOW_NAME = "Reprojection Verification"
PROJ_COLOR = (0, 0, 255)     # 再投影点の色（赤）[BGR]
ANNO_COLOR = (0, 255, 0)     # アノテーション済み点の色（緑）[BGR]
PROJ_RADIUS = 2
ANNO_RADIUS = 5


# ---------------------------------------------------------------------------
# データ読み込みユーティリティ
# ---------------------------------------------------------------------------

def load_transform(path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    transform_matrix.json を読み込み、rvec, tvec, calibration_frame_index を返す。

    Parameters
    ----------
    path : str
        transform_matrix.json のパス

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, int]
        (rvec [3x1], tvec [3x1], calibration_frame_index)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rvec = np.array(data["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.array(data["tvec"], dtype=np.float64).reshape(3, 1)

    # 旧形式（frame_index）との後方互換
    calib_idx = int(
        data.get("calibration_frame_index", data.get("frame_index", 0))
    )

    print(f"[変換行列] calibration_frame_index={calib_idx}")
    print(f"  rvec: {rvec.ravel()}")
    print(f"  tvec: {tvec.ravel()}")

    return rvec, tvec, calib_idx


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
    """
    p = Path(path)
    if not p.exists():
        print(f"[情報] {path} が見つかりません。アノテーション点の表示をスキップします。")
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


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
    annotated_points: Optional[Dict],
    calib_frame_index: int,
) -> np.ndarray:
    """
    3D 点をヒートマップ画像に再投影して描画した表示用画像を返す。

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
        カメラ行列
    dist_coeffs : np.ndarray
        歪み係数
    annotated_points : Optional[Dict]
        annotated_points.json の内容（None の場合は表示しない）
    calib_frame_index : int
        3D 座標の取得元フレームインデックス（表示用）

    Returns
    -------
    np.ndarray
        オーバーレイ済み BGR 画像（表示倍率適用後）
    """
    img = base_frame.copy()
    h, w = img.shape[:2]

    # --- 全顔メッシュ点の再投影 ---
    if len(points_3d) > 0:
        projected, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec, tvec, camera_matrix, dist_coeffs,
        )
        projected = projected.reshape(-1, 2)

        for pt in projected:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), PROJ_RADIUS, PROJ_COLOR, -1)

    # --- アノテーション済み 7 点を緑で重ね描き ---
    if annotated_points is not None:
        for lm in annotated_points.get("landmarks", []):
            u, v = lm["point_2d"]
            x, y = int(round(u)), int(round(v))
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img, (x, y), ANNO_RADIUS, ANNO_COLOR, 2)
            # ラベル表示
            cv2.putText(img, lm["name"], (x + 6, y - 4),
                        FONT, 0.32, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, lm["name"], (x + 6, y - 4),
                        FONT, 0.32, ANNO_COLOR, 1, cv2.LINE_AA)

    # --- 情報テキスト ---
    n_proj = len(points_3d)
    info1 = (
        f"3D src: frame {calib_frame_index}  |  "
        f"Projected: {n_proj} pts"
    )
    info2 = "Red: projected  Green: annotated  [q] quit"

    cv2.rectangle(img, (0, 0), (w, 40), TEXT_BG_COLOR, -1)
    cv2.putText(img, info1, (4, 14), FONT, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(img, info2, (4, 33), FONT, 0.36, (180, 180, 100), 1, cv2.LINE_AA)

    # 表示倍率適用
    disp = cv2.resize(
        img,
        (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
        interpolation=cv2.INTER_NEAREST,
    )
    return disp


# ---------------------------------------------------------------------------
# メイン検証クラス
# ---------------------------------------------------------------------------

class ReprojectionVerifier:
    """
    rvec / tvec を使って顔メッシュ全点をヒートマップ上に再投影して確認するクラス。

    Attributes
    ----------
    config : Dict
        設定辞書
    rvec : np.ndarray
        回転ベクトル [3x1]
    tvec : np.ndarray
        並進ベクトル [3x1]
    calib_frame_index : int
        PnP 計算に使用した calibration_frame_index
    camera_matrix : np.ndarray
        カメラ行列 [3x3]
    dist_coeffs : np.ndarray
        歪み係数 [5x1]
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

        print("[初期化] landmark.csv を読み込み中...")
        self.landmark_df = load_landmark_csv(self._resolve(paths["landmark"]))

        print("[初期化] calibration.json を読み込み中...")
        self.camera_matrix, self.dist_coeffs = load_calibration(
            self._resolve(paths["calibration"])
        )

        print("[初期化] transform_matrix.json を読み込み中...")
        self.rvec, self.tvec, self.calib_frame_index = load_transform(
            self._resolve(paths["output_transform"])
        )

        self.annotated_points = load_annotated_points(
            self._resolve(paths["output_points"])
        )

        self.events_path = self._resolve(paths["events"])
        self.ef_cfg = config["event_frame"]

        self._heatmap_bgr: Optional[np.ndarray] = None

        print("[初期化] 完了")

    def _ensure_heatmap(self) -> None:
        """ヒートマップを遅延生成する（1 度だけ生成）。"""
        if self._heatmap_bgr is not None:
            return
        print("[ヒートマップ生成] events.csv を読み込み中（しばらくお待ちください）...")
        self._heatmap_bgr = build_annotation_heatmap(
            events_path=self.events_path,
            ef_cfg=self.ef_cfg,
            colormap=cv2.COLORMAP_INFERNO,
        )
        print(f"[ヒートマップ生成] 完了: shape={self._heatmap_bgr.shape}")

    def run(self) -> None:
        """
        GUI を起動し、再投影結果を表示する。
        """
        self._ensure_heatmap()

        # calibration_frame_index の全ランドマーク 3D 点を取得
        points_3d, landmark_ids = get_all_3d_points_for_frame(
            self.landmark_df, self.calib_frame_index
        )
        print(
            f"[再投影] frame={self.calib_frame_index}: {len(points_3d)} 点を投影"
        )

        display = project_and_draw(
            base_frame=self._heatmap_bgr,
            points_3d=points_3d,
            rvec=self.rvec,
            tvec=self.tvec,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            annotated_points=self.annotated_points,
            calib_frame_index=self.calib_frame_index,
        )

        cv2.namedWindow(VERIFY_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        print("\n=== Reprojection Verifier ===")
        print("  q / Esc: 終了")
        print("=============================\n")

        while True:
            cv2.imshow(VERIFY_WINDOW_NAME, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                try:
                    if cv2.getWindowProperty(VERIFY_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("[終了] ウィンドウが閉じられました。")
                        break
                except cv2.error:
                    break
                continue

            elif key in (ord("q"), 27):
                print("[終了] Verifier を終了します。")
                break

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    """メインエントリーポイント。"""
    parser = argparse.ArgumentParser(
        description=(
            "Reprojection Verifier: PnP 変換行列を使って顔メッシュを"
            "ヒートマップに再投影して目視確認する。"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="config.json のパス (デフォルト: config.json)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[エラー] config.json が見つかりません: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    config_dir = config_path.parent

    verifier = ReprojectionVerifier(config, config_dir)
    verifier.run()


if __name__ == "__main__":
    main()
