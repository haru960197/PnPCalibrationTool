"""
main_pnp_solver.py
------------------
PnP（Perspective-n-Point）問題を解くメインツール。

【新アノテーションフロー】
  Step 1 : events.csv の絶対時刻範囲（start_delay_us ～ +accumulation_time_us）を
           空間積算したヒートマップ画像を生成する。
           （sync_params による厳密な時刻同期は使用しない）
  Step 2 : OpenCV GUI でヒートマップ上にランドマーク 7 点をマウスクリックでアノテーション。
  Step 3 : config.json の calibration_frame_index で指定した RGB フレームの
           landmark.csv から 3D 座標を取得して 2D/3D ペアを構成する。
  Step 4 : cv2.solvePnP で rvec / tvec を求解し、JSON に保存する。

使い方:
  python main_pnp_solver.py [--config CONFIG]

キーボード操作 (GUI):
  u      : 直前のクリックを取り消す（Undo）
  r      : アノテーションをリセット
  s      : PnP を計算して保存
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


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
WINDOW_NAME = "PnP Solver - Heatmap Annotation"
CLICK_RADIUS = 6
CLICK_COLOR_DONE = (0, 255, 0)       # 記録済み点の色 (BGR: 緑)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_SCALE = 2                    # 320px → 640px 表示


# ---------------------------------------------------------------------------
# データ読み込みユーティリティ
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """
    config.json を読み込んで辞書として返す。

    Parameters
    ----------
    config_path : str
        config.json のファイルパス

    Returns
    -------
    Dict
        設定辞書
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_calibration(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    calibration.json を読み込み、カメラ行列と歪み係数を返す。

    Parameters
    ----------
    path : str
        calibration.json のパス

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (camera_matrix [3x3], dist_coeffs [5x1])
    """
    with open(path, "r", encoding="utf-8") as f:
        cal = json.load(f)
    camera_matrix = np.array(cal["intrinsics"], dtype=np.float64)
    dist_coeffs = np.array(cal["distortion"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def load_landmark_csv(path: str) -> pd.DataFrame:
    """
    landmark.csv を読み込む。

    カラム: frame_index, face_index, landmark_index, x_norm, y_norm, z_norm

    Parameters
    ----------
    path : str
        landmark.csv のパス

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(
        path,
        dtype={
            "frame_index": int,
            "face_index": int,
            "landmark_index": int,
            "x_norm": np.float64,
            "y_norm": np.float64,
            "z_norm": np.float64,
        },
    )
    return df


def get_3d_points_for_frame(
    landmark_df: pd.DataFrame,
    frame_index: int,
    landmark_ids: List[int],
) -> Optional[np.ndarray]:
    """
    指定フレームと指定ランドマーク ID に対応する 3D 座標を取得する。

    Parameters
    ----------
    landmark_df : pd.DataFrame
        landmark.csv の DataFrame
    frame_index : int
        対象フレームインデックス（calibration_frame_index）
    landmark_ids : List[int]
        対象ランドマーク ID のリスト

    Returns
    -------
    Optional[np.ndarray], shape (N, 3)
        3D 座標配列。取得できない場合は None。
    """
    frame_df = landmark_df[landmark_df["frame_index"] == frame_index]
    if frame_df.empty:
        print(f"  [警告] frame_index={frame_index} のデータが landmark.csv に存在しません。")
        return None

    rows = []
    for lid in landmark_ids:
        row = frame_df[frame_df["landmark_index"] == lid]
        if row.empty:
            print(f"  [警告] frame_index={frame_index}, landmark_index={lid} が見つかりません。")
            return None
        rows.append(row[["x_norm", "y_norm", "z_norm"]].values[0])

    return np.array(rows, dtype=np.float64)


def get_all_3d_points_for_frame(
    landmark_df: pd.DataFrame,
    frame_index: int,
) -> Tuple[np.ndarray, List[int]]:
    """
    指定フレームの全ランドマークの 3D 座標を取得する。（verify_reprojection 用に公開）

    Parameters
    ----------
    landmark_df : pd.DataFrame
        landmark.csv の DataFrame
    frame_index : int
        対象フレームインデックス

    Returns
    -------
    Tuple[np.ndarray, List[int]]
        (points_3d [N x 3], landmark_ids [N])
    """
    frame_df = landmark_df[landmark_df["frame_index"] == frame_index].copy()
    frame_df = frame_df.sort_values("landmark_index")
    if frame_df.empty:
        return np.empty((0, 3), dtype=np.float64), []
    points_3d = frame_df[["x_norm", "y_norm", "z_norm"]].values.astype(np.float64)
    landmark_ids = frame_df["landmark_index"].tolist()
    return points_3d, landmark_ids


# ---------------------------------------------------------------------------
# ヒートマップ生成
# ---------------------------------------------------------------------------

def build_annotation_heatmap(
    events_path: str,
    ef_cfg: Dict,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    events.csv からアノテーション用のヒートマップ BGR 画像を生成する。

    処理:
        1. start_delay_us ～ +accumulation_time_us の範囲でイベントを空間積算
        2. log1p 変換
        3. clip_percentile パーセンタイルでクリッピング
        4. 0-255 線形スケール → uint8 グレースケール
        5. カラーマップを適用して BGR 画像として返す

    Parameters
    ----------
    events_path : str
        events.csv のパス
    ef_cfg : Dict
        config.json の "event_frame" セクション辞書
    colormap : int
        OpenCV カラーマップ定数（-1 でグレースケール）

    Returns
    -------
    np.ndarray, shape (height, width, 3), dtype=uint8
        BGR ヒートマップ画像
    """
    width = int(ef_cfg["width"])
    height = int(ef_cfg["height"])
    start_delay_us = float(ef_cfg["start_delay_us"])
    accumulation_time_us = float(ef_cfg["accumulation_time_us"])
    clip_percentile = float(ef_cfg.get("clip_percentile", 99.5))

    print(
        f"[ヒートマップ] start_delay={start_delay_us:.0f} µs, "
        f"accumulation={accumulation_time_us:.0f} µs, "
        f"clip_percentile={clip_percentile}%"
    )

    raw = build_event_heatmap(
        filepath=events_path,
        start_delay_us=start_delay_us,
        accumulation_time_us=accumulation_time_us,
        width=width,
        height=height,
    )
    gray = normalize_to_uint8(raw, clip_percentile=clip_percentile)
    bgr = heatmap_to_bgr(gray, colormap=colormap)
    return bgr


# ---------------------------------------------------------------------------
# GUI アノテーター
# ---------------------------------------------------------------------------

class HeatmapAnnotator:
    """
    ヒートマップ画像上でランドマークを手動アノテーションし、PnP 問題を解くクラス。

    旧実装との主な相違点:
    - フレーム切り替え不要: ヒートマップは単一の積算画像（キャリブレーション期間全体）
    - 3D 座標の取得元: config.json の calibration_frame_index で固定した RGB フレーム
    - sync_params (A, B) の利用なし

    Attributes
    ----------
    config : Dict
        設定辞書
    landmark_df : pd.DataFrame
        landmark.csv の DataFrame
    camera_matrix : np.ndarray
        イベントカメラのカメラ行列 [3x3]
    dist_coeffs : np.ndarray
        イベントカメラの歪み係数 [5x1]
    calibration_frame_index : int
        3D 座標取得に使用する RGB フレームインデックス
    target_landmarks : List[Dict]
        アノテーション対象のランドマーク定義リスト
    heatmap_bgr : np.ndarray
        生成済みのヒートマップ BGR 画像（キャッシュ）
    clicked : List[Tuple[int, int]]
        クリック済み座標リスト
    """

    def __init__(self, config: Dict, config_dir: Path):
        """
        Parameters
        ----------
        config : Dict
            config.json を解析した辞書
        config_dir : Path
            config.json が置かれているディレクトリ（相対パス解決に使用）
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

        self.events_path = self._resolve(paths["events"])
        self.output_transform_path = self._resolve(paths["output_transform"])
        self.output_points_path = self._resolve(paths["output_points"])

        # calibration_frame_index: 3D 座標の取得元フレーム
        self.calibration_frame_index = int(
            config["rgb_frame"]["calibration_frame_index"]
        )
        print(
            f"[初期化] calibration_frame_index = {self.calibration_frame_index} "
            f"(3D 座標の取得元)"
        )

        self.ef_cfg = config["event_frame"]
        self.width = int(self.ef_cfg["width"])
        self.height = int(self.ef_cfg["height"])

        self.target_landmarks = config["target_landmarks"]
        self.n_points = len(self.target_landmarks)

        # クリック座標の保持リスト
        self.clicked: List[Tuple[int, int]] = []

        # ヒートマップ（起動時に 1 度だけ生成）
        self.heatmap_bgr: Optional[np.ndarray] = None

        # 再描画フラグ
        self._need_redraw = True

        print("[初期化] 完了")

    # ------------------------------------------------------------------ #
    # ヒートマップ生成
    # ------------------------------------------------------------------ #

    def _ensure_heatmap(self) -> None:
        """
        ヒートマップをまだ生成していない場合にのみ生成する（遅延初期化）。
        """
        if self.heatmap_bgr is not None:
            return
        print("[ヒートマップ生成] events.csv を読み込み中（しばらくお待ちください）...")
        self.heatmap_bgr = build_annotation_heatmap(
            events_path=self.events_path,
            ef_cfg=self.ef_cfg,
            colormap=cv2.COLORMAP_INFERNO,
        )
        print(
            f"[ヒートマップ生成] 完了: shape={self.heatmap_bgr.shape}, "
            f"dtype={self.heatmap_bgr.dtype}"
        )

    # ------------------------------------------------------------------ #
    # 描画
    # ------------------------------------------------------------------ #

    def _draw_overlay(self) -> np.ndarray:
        """
        ヒートマップにアノテーション用のオーバーレイを描画した表示用画像を返す。

        Returns
        -------
        np.ndarray, shape (height*DISPLAY_SCALE, width*DISPLAY_SCALE, 3)
            表示倍率適用済みの BGR 画像
        """
        img = self.heatmap_bgr.copy()
        h, w = img.shape[:2]

        # 記録済み点を描画
        for i, (u, v) in enumerate(self.clicked):
            name = self.target_landmarks[i]["name"]
            cv2.circle(img, (u, v), CLICK_RADIUS, CLICK_COLOR_DONE, -1)
            # 白のアウトライン付きテキストで視認性向上
            cv2.putText(img, name, (u + 8, v - 4), FONT, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img, name, (u + 8, v - 4), FONT, 0.35, CLICK_COLOR_DONE, 1, cv2.LINE_AA)

        # ガイドテキスト
        n_done = len(self.clicked)
        if n_done < self.n_points:
            next_lm = self.target_landmarks[n_done]
            guide_text = (
                f"[{n_done + 1}/{self.n_points}] Click: {next_lm['name']}"
            )
            guide_color = (0, 200, 255)
        else:
            guide_text = "All points annotated! [s] Save / [r] Reset"
            guide_color = (0, 255, 128)

        # 上部情報帯
        calib_info = (
            f"Heatmap  |  3D src: frame {self.calibration_frame_index}  |  "
            f"[u]undo [r]reset [s]save [q]quit"
        )
        cv2.rectangle(img, (0, 0), (w, 40), TEXT_BG_COLOR, -1)
        cv2.putText(img, calib_info, (4, 14), FONT, 0.36, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, guide_text, (4, 33), FONT, 0.44, guide_color, 1, cv2.LINE_AA)

        # 表示倍率を適用
        disp = cv2.resize(
            img,
            (w * DISPLAY_SCALE, h * DISPLAY_SCALE),
            interpolation=cv2.INTER_NEAREST,
        )
        return disp

    # ------------------------------------------------------------------ #
    # マウスコールバック
    # ------------------------------------------------------------------ #

    def _on_mouse(self, event, x, y, flags, param):
        """
        マウス左クリックで現在のターゲット座標を記録する。
        表示倍率の逆変換を行って元画像座標に変換する。
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        orig_x = int(x / DISPLAY_SCALE)
        orig_y = int(y / DISPLAY_SCALE)

        if len(self.clicked) < self.n_points:
            self.clicked.append((orig_x, orig_y))
            name = self.target_landmarks[len(self.clicked) - 1]["name"]
            print(
                f"  クリック [{len(self.clicked)}/{self.n_points}] "
                f"{name}: ({orig_x}, {orig_y})"
            )
            self._need_redraw = True

    # ------------------------------------------------------------------ #
    # PnP 計算と保存
    # ------------------------------------------------------------------ #

    def _solve_and_save(self) -> bool:
        """
        アノテーション済み 2D 点と calibration_frame_index の 3D 点から
        PnP を解いて結果を JSON に保存する。

        Returns
        -------
        bool
            成功した場合 True
        """
        if len(self.clicked) < self.n_points:
            print(
                f"[エラー] アノテーションが {self.n_points} 点に満たないため "
                f"PnP を実行できません。({len(self.clicked)} / {self.n_points} 点)"
            )
            return False

        # 2D 点（ヒートマップ上のピクセル座標）
        points_2d = np.array(self.clicked[: self.n_points], dtype=np.float64)

        # 3D 点（calibration_frame_index の landmark.csv から取得）
        landmark_ids = [lm["id"] for lm in self.target_landmarks]
        points_3d = get_3d_points_for_frame(
            self.landmark_df,
            self.calibration_frame_index,
            landmark_ids,
        )
        if points_3d is None:
            print("[エラー] landmark.csv から 3D 座標を取得できませんでした。")
            return False

        print(f"[PnP] 2D 点 (ヒートマップ座標):\n{points_2d}")
        print(f"[PnP] 3D 点 (MediaPipe 正規化座標, frame={self.calibration_frame_index}):\n{points_3d}")
        print(f"[PnP] カメラ行列:\n{self.camera_matrix}")
        print(f"[PnP] 歪み係数: {self.dist_coeffs.ravel()}")

        # PnP 求解
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print("[エラー] cv2.solvePnP が失敗しました。")
            return False

        print(f"[PnP 結果] rvec: {rvec.ravel()}")
        print(f"[PnP 結果] tvec: {tvec.ravel()}")

        # 再投影誤差
        projected_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        projected_2d = projected_2d.reshape(-1, 2)
        errors = np.linalg.norm(points_2d - projected_2d, axis=1)
        print(f"[PnP] 各点の再投影誤差 [px]: {errors.round(2)}")
        print(f"[PnP] 平均再投影誤差: {errors.mean():.3f} px")

        # 出力ディレクトリの作成
        Path(self.output_transform_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_points_path).parent.mkdir(parents=True, exist_ok=True)

        # transform_matrix.json の保存
        transform_data = {
            "calibration_frame_index": self.calibration_frame_index,
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
            "reprojection_errors_px": errors.tolist(),
            "mean_reprojection_error_px": float(errors.mean()),
            "heatmap_params": {
                "start_delay_us": float(self.ef_cfg["start_delay_us"]),
                "accumulation_time_us": float(self.ef_cfg["accumulation_time_us"]),
                "clip_percentile": float(self.ef_cfg.get("clip_percentile", 99.5)),
            },
        }
        with open(self.output_transform_path, "w", encoding="utf-8") as f:
            json.dump(transform_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] {self.output_transform_path}")

        # annotated_points.json の保存
        points_data = {
            "calibration_frame_index": self.calibration_frame_index,
            "landmarks": [
                {
                    "id": self.target_landmarks[i]["id"],
                    "name": self.target_landmarks[i]["name"],
                    "point_2d": list(points_2d[i]),
                    "point_3d": list(points_3d[i]),
                    "reprojected_2d": list(projected_2d[i]),
                    "error_px": float(errors[i]),
                }
                for i in range(self.n_points)
            ],
        }
        with open(self.output_points_path, "w", encoding="utf-8") as f:
            json.dump(points_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] {self.output_points_path}")

        return True

    # ------------------------------------------------------------------ #
    # メインループ
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """
        GUI メインループを起動する。

        ヒートマップを生成してウィンドウに表示し、
        ユーザーのキー操作・マウスクリックを処理する。
        """
        # ヒートマップ生成（大規模ファイルのため起動直後に実行）
        self._ensure_heatmap()

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        print("\n=== PnP Solver (ヒートマップモード) 操作ガイド ===")
        print("  マウス左クリック : ランドマークを順番に指定")
        print("  u               : 直前のクリックを取り消し (Undo)")
        print("  r               : アノテーションをリセット")
        print("  s               : PnP を計算して output/ に保存")
        print("  q / Esc         : 終了")
        print(f"  3D 座標の取得元  : RGB frame {self.calibration_frame_index}")
        print("=================================================\n")

        while True:
            if self._need_redraw:
                display = self._draw_overlay()
                cv2.imshow(WINDOW_NAME, display)
                self._need_redraw = False

            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                # タイムアウト → ウィンドウ閉じ確認
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        print("[終了] ウィンドウが閉じられました。")
                        break
                except cv2.error:
                    break
                continue

            # u : Undo
            elif key == ord("u"):
                if self.clicked:
                    removed = self.clicked.pop()
                    print(f"  [Undo] 削除: {removed}")
                    self._need_redraw = True
                else:
                    print("[情報] 取り消せる点がありません。")

            # r : リセット
            elif key == ord("r"):
                self.clicked.clear()
                print("  [リセット] アノテーションをリセットしました。")
                self._need_redraw = True

            # s : 保存
            elif key == ord("s"):
                if self._solve_and_save():
                    print("[成功] PnP 計算・保存が完了しました。")
                else:
                    print("[失敗] PnP 計算に失敗しました。")
                self._need_redraw = True

            # q / Esc : 終了
            elif key in (ord("q"), 27):
                print("[終了] GUI を終了します。")
                break

        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    """メインエントリーポイント。コマンドライン引数を解析して GUI を起動する。"""
    parser = argparse.ArgumentParser(
        description=(
            "PnP Solver (ヒートマップモード): "
            "キャリブレーション期間のイベント蓄積ヒートマップ上でランドマークをアノテーションし、"
            "変換行列を求解する。"
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

    annotator = HeatmapAnnotator(config, config_dir)
    annotator.run()


if __name__ == "__main__":
    main()
