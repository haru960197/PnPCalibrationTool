"""
main_pnp_solver.py
------------------
PnP（Perspective-n-Point）問題を解くメインツール。

処理フロー:
  Step 1 : sync_log.csv と sync_params.json を使ってイベントフレームを生成
  Step 2 : OpenCV GUI でユーザーが 7 点のランドマークを手動アノテーション
  Step 3 : landmark.csv から対応する 3D 座標を取得し cv2.solvePnP を実行
  Step 4 : rvec / tvec を JSON に保存

使い方:
  python main_pnp_solver.py [--config CONFIG] [--frame FRAME_INDEX]

キーボード操作 (GUI):
  n / →  : 次のフレームへ
  p / ←  : 前のフレームへ
  u      : 直前のクリックを取り消す（Undo）
  r      : 現フレームのアノテーションをリセット
  s      : 現フレームで PnP を計算して保存
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


# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
WINDOW_NAME = "PnP Solver - Event Frame Annotation"
CLICK_RADIUS = 6          # クリック点を描画する円の半径 [pixels]
CLICK_COLOR_DONE = (0, 255, 0)    # 記録済み点の色 (BGR)
CLICK_COLOR_CURRENT = (0, 165, 255)  # 現在ターゲット点の色 (BGR)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (30, 30, 30)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DISPLAY_SCALE = 2          # 表示倍率（小さい画像を見やすくするため）


# ---------------------------------------------------------------------------
# ユーティリティ関数
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


def load_sync_log(path: str) -> pd.DataFrame:
    """
    sync_log.csv を読み込む。

    カラム: frame_index, timestamp_ms, led_status

    Parameters
    ----------
    path : str
        sync_log.csv のパス

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path, dtype={"frame_index": int, "timestamp_ms": float})
    return df.set_index("frame_index")


def load_sync_params(path: str) -> Tuple[float, float]:
    """
    sync_params.json を読み込み (A, B) を返す。

    関係式: t_rgb [ms] = A * t_event [µs] + B

    Parameters
    ----------
    path : str
        sync_params.json のパス

    Returns
    -------
    Tuple[float, float]
        (A, B)
    """
    with open(path, "r") as f:
        params = json.load(f)
    return float(params["A"]), float(params["B"])


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
        (camera_matrix [3x3], dist_coeffs [5,])
    """
    with open(path, "r") as f:
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


def rgb_timestamp_to_event_timestamp(
    timestamp_ms: float, A: float, B: float
) -> float:
    """
    RGB カメラのタイムスタンプ [ms] をイベントカメラのタイムスタンプ [µs] へ変換する。

    変換式 (逆算): t_event = (t_rgb - B) / A

    Parameters
    ----------
    timestamp_ms : float
        RGB カメラのタイムスタンプ [ms]
    A : float
        同期パラメータ A
    B : float
        同期パラメータ B

    Returns
    -------
    float
        イベントカメラのタイムスタンプ [µs]
    """
    return (timestamp_ms - B) / A


def get_event_frame_for_rgb_frame(
    frame_index: int,
    sync_log: pd.DataFrame,
    A: float,
    B: float,
    events_path: str,
    width: int,
    height: int,
    integration_time_ms: float,
) -> np.ndarray:
    """
    指定した RGB フレームインデックスに対応するイベントフレームを生成する。

    Parameters
    ----------
    frame_index : int
        RGB フレームのインデックス
    sync_log : pd.DataFrame
        sync_log.csv を読み込んだ DataFrame（frame_index がインデックス）
    A : float
        同期パラメータ A
    B : float
        同期パラメータ B
    events_path : str
        events.csv のパス
    width : int
        出力フレームの幅 [pixels]
    height : int
        出力フレームの高さ [pixels]
    integration_time_ms : float
        積分時間 [ms]（この時間幅のイベントを収集）

    Returns
    -------
    np.ndarray, shape (height, width, 3)
        BGR イベントフレーム画像
    """
    if frame_index not in sync_log.index:
        raise ValueError(f"frame_index={frame_index} は sync_log に存在しません。")

    timestamp_ms = float(sync_log.loc[frame_index, "timestamp_ms"])
    t_target_us = rgb_timestamp_to_event_timestamp(timestamp_ms, A, B)

    # 積分時間の半分 [µs]
    half_window_us = (integration_time_ms / 2.0) * 1000.0
    t_start = t_target_us - half_window_us
    t_end = t_target_us + half_window_us

    events = load_events_in_range(events_path, t_start, t_end)
    return build_event_frame(events, width, height)


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
        対象フレームインデックス
    landmark_ids : List[int]
        対象ランドマーク ID のリスト（config.json の target_landmarks の id）

    Returns
    -------
    Optional[np.ndarray], shape (N, 3)
        3D 座標配列。取得できない場合は None。
    """
    frame_df = landmark_df[landmark_df["frame_index"] == frame_index]
    if frame_df.empty:
        return None

    rows = []
    for lid in landmark_ids:
        row = frame_df[frame_df["landmark_index"] == lid]
        if row.empty:
            print(f"  [警告] frame_index={frame_index}, landmark_index={lid} が見つかりません。")
            return None
        rows.append(row[["x_norm", "y_norm", "z_norm"]].values[0])

    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# GUI アノテーター
# ---------------------------------------------------------------------------
class EventFrameAnnotator:
    """
    OpenCV GUI を使ってイベントフレーム上でランドマークを手動アノテーションするクラス。

    ユーザーは config.json で指定された N 個のランドマーク（例: 7 点）を
    順番にマウスクリックで指定する。

    Attributes
    ----------
    config : Dict
        設定辞書
    sync_log : pd.DataFrame
        sync_log.csv の DataFrame
    A, B : float
        同期パラメータ
    landmark_df : pd.DataFrame
        landmark.csv の DataFrame
    camera_matrix : np.ndarray
        イベントカメラの内部パラメータ行列 [3x3]
    dist_coeffs : np.ndarray
        イベントカメラの歪み係数 [5x1]
    frame_indices : List[int]
        利用可能な RGB フレームインデックスの一覧
    target_landmarks : List[Dict]
        アノテーション対象のランドマーク定義リスト
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

        paths = config["paths"]
        self._resolve = lambda p: str(config_dir / p)

        # データ読み込み
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

        self.events_path = self._resolve(paths["events"])
        self.output_transform_path = self._resolve(paths["output_transform"])
        self.output_points_path = self._resolve(paths["output_points"])

        ef_cfg = config["event_frame"]
        self.width = int(ef_cfg["width"])
        self.height = int(ef_cfg["height"])
        self.integration_time_ms = float(ef_cfg["integration_time_ms"])

        self.target_landmarks = config["target_landmarks"]
        self.n_points = len(self.target_landmarks)

        # フレームインデックス一覧
        self.frame_indices = sorted(self.sync_log.index.tolist())
        self.current_frame_pos = 0  # frame_indices 内のポインタ

        # アノテーション状態
        # { frame_index: [(u, v), ...] } — フレームごとにクリック済み座標を保持
        self.annotations: Dict[int, List[Tuple[int, int]]] = {}

        # イベントフレームキャッシュ（重複生成を防ぐ）
        self._frame_cache: Dict[int, np.ndarray] = {}

        print("[初期化] 完了")

    # ------------------------------------------------------------------ #
    # イベントフレーム取得
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # 描画
    # ------------------------------------------------------------------ #

    def _draw_overlay(
        self,
        base: np.ndarray,
        frame_index: int,
        clicked: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        イベントフレームにアノテーション用のオーバーレイを描画する。

        Parameters
        ----------
        base : np.ndarray
            描画対象のベース BGR 画像（コピーして使用）
        frame_index : int
            現在の RGB フレームインデックス
        clicked : List[Tuple[int, int]]
            現在クリック済みの座標リスト

        Returns
        -------
        np.ndarray
            オーバーレイ済み BGR 画像（表示倍率適用後）
        """
        img = base.copy()
        h, w = img.shape[:2]

        # 記録済み点を描画
        for i, (u, v) in enumerate(clicked):
            name = self.target_landmarks[i]["name"]
            cv2.circle(img, (u, v), CLICK_RADIUS, CLICK_COLOR_DONE, -1)
            cv2.putText(
                img, name, (u + 8, v - 4),
                FONT, 0.35, CLICK_COLOR_DONE, 1, cv2.LINE_AA
            )

        # 次にクリックすべきランドマーク名を上部に表示
        n_done = len(clicked)
        if n_done < self.n_points:
            next_lm = self.target_landmarks[n_done]
            guide_text = f"[{n_done + 1}/{self.n_points}] クリック: {next_lm['name']}"
        else:
            guide_text = "全点完了! [s] で保存 / [r] でリセット"

        # フレーム情報
        info_text = (
            f"Frame: {frame_index}  "
            f"[n]次 [p]前 [u]undo [r]reset [s]save [q]quit"
        )

        # テキスト背景帯
        cv2.rectangle(img, (0, 0), (w, 38), TEXT_BG_COLOR, -1)
        cv2.putText(img, info_text, (4, 14), FONT, 0.38, TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(img, guide_text, (4, 32), FONT, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

        # 表示倍率を適用
        disp_w = w * DISPLAY_SCALE
        disp_h = h * DISPLAY_SCALE
        img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        return img

    # ------------------------------------------------------------------ #
    # マウスコールバック
    # ------------------------------------------------------------------ #

    def _on_mouse(self, event, x, y, flags, param):
        """
        マウスクリックイベントのコールバック。

        左クリックで現在のターゲット座標を記録する。
        表示倍率を考慮して元画像の座標に変換する。
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 表示倍率の逆変換
        orig_x = int(x / DISPLAY_SCALE)
        orig_y = int(y / DISPLAY_SCALE)

        frame_index = self.frame_indices[self.current_frame_pos]
        clicked = self.annotations.setdefault(frame_index, [])

        if len(clicked) < self.n_points:
            clicked.append((orig_x, orig_y))
            name = self.target_landmarks[len(clicked) - 1]["name"]
            print(f"  クリック [{len(clicked)}/{self.n_points}] {name}: ({orig_x}, {orig_y})")
            self._need_redraw = True

    # ------------------------------------------------------------------ #
    # PnP 計算と保存
    # ------------------------------------------------------------------ #

    def _solve_and_save(self, frame_index: int) -> bool:
        """
        指定フレームのアノテーションから PnP を解いて結果を保存する。

        Parameters
        ----------
        frame_index : int
            対象の RGB フレームインデックス

        Returns
        -------
        bool
            成功した場合 True
        """
        clicked = self.annotations.get(frame_index, [])

        if len(clicked) < self.n_points:
            print(f"[エラー] アノテーションが {self.n_points} 点に満たないため PnP を実行できません。")
            return False

        # 2D 点（イベントフレーム座標）
        points_2d = np.array(clicked[: self.n_points], dtype=np.float64)

        # 3D 点（MediaPipe 正規化座標）
        landmark_ids = [lm["id"] for lm in self.target_landmarks]
        points_3d = get_3d_points_for_frame(
            self.landmark_df, frame_index, landmark_ids
        )

        if points_3d is None:
            print("[エラー] landmark.csv から 3D 座標を取得できませんでした。")
            return False

        print(f"[PnP] 2D 点:\n{points_2d}")
        print(f"[PnP] 3D 点:\n{points_3d}")
        print(f"[PnP] カメラ行列:\n{self.camera_matrix}")
        print(f"[PnP] 歪み係数:\n{self.dist_coeffs.ravel()}")

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

        # 再投影誤差を計算して表示
        projected_2d, _ = cv2.projectPoints(
            points_3d, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        projected_2d = projected_2d.reshape(-1, 2)
        errors = np.linalg.norm(points_2d - projected_2d, axis=1)
        print(f"[PnP] 再投影誤差 (px): {errors}")
        print(f"[PnP] 平均再投影誤差: {errors.mean():.3f} px")

        # 出力ディレクトリを作成
        Path(self.output_transform_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_points_path).parent.mkdir(parents=True, exist_ok=True)

        # transform_matrix.json に保存
        transform_data = {
            "frame_index": frame_index,
            "rvec": rvec.ravel().tolist(),
            "tvec": tvec.ravel().tolist(),
            "reprojection_errors_px": errors.tolist(),
            "mean_reprojection_error_px": float(errors.mean()),
        }
        with open(self.output_transform_path, "w", encoding="utf-8") as f:
            json.dump(transform_data, f, ensure_ascii=False, indent=2)
        print(f"[保存] {self.output_transform_path}")

        # annotated_points.json に保存
        landmark_names = [lm["name"] for lm in self.target_landmarks]
        points_data = {
            "frame_index": frame_index,
            "landmarks": [
                {
                    "id": self.target_landmarks[i]["id"],
                    "name": landmark_names[i],
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

    def run(self, initial_frame_index: Optional[int] = None):
        """
        GUI メインループを起動する。

        Parameters
        ----------
        initial_frame_index : Optional[int]
            起動時に表示する RGB フレームインデックス。
            None の場合は最初のフレームから開始。
        """
        if not self.frame_indices:
            print("[エラー] 利用可能なフレームがありません。")
            return

        # 初期フレーム位置を設定
        if initial_frame_index is not None and initial_frame_index in self.frame_indices:
            self.current_frame_pos = self.frame_indices.index(initial_frame_index)
        else:
            self.current_frame_pos = 0

        # ウィンドウ作成
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self._need_redraw = True

        print("\n=== PnP Solver GUI 操作ガイド ===")
        print("  n / →  : 次のフレームへ")
        print("  p / ←  : 前のフレームへ")
        print("  u      : 直前のクリックを取り消し (Undo)")
        print("  r      : 現フレームのアノテーションをリセット")
        print("  s      : PnP を計算して保存")
        print("  q / Esc: 終了")
        print("=================================\n")

        while True:
            frame_index = self.frame_indices[self.current_frame_pos]

            if self._need_redraw:
                base_frame = self._get_event_frame(frame_index)
                clicked = self.annotations.get(frame_index, [])
                display = self._draw_overlay(base_frame, frame_index, clicked)
                cv2.imshow(WINDOW_NAME, display)
                self._need_redraw = False

            # キー入力待ち（30ms）
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                # タイムアウト（入力なし）→ ループ継続
                # ウィンドウが閉じられたか確認
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[終了] ウィンドウが閉じられました。")
                    break
                continue

            # n / → : 次フレーム
            elif key in (ord("n"), 83):  # 83 = → (右矢印)
                if self.current_frame_pos < len(self.frame_indices) - 1:
                    self.current_frame_pos += 1
                    self._need_redraw = True
                else:
                    print("[情報] 最後のフレームです。")

            # p / ← : 前フレーム
            elif key in (ord("p"), 81):  # 81 = ← (左矢印)
                if self.current_frame_pos > 0:
                    self.current_frame_pos -= 1
                    self._need_redraw = True
                else:
                    print("[情報] 最初のフレームです。")

            # u : Undo
            elif key == ord("u"):
                clicked = self.annotations.get(frame_index, [])
                if clicked:
                    removed = clicked.pop()
                    print(f"  [Undo] 削除: {removed}")
                    self._need_redraw = True
                else:
                    print("[情報] 取り消せる点がありません。")

            # r : リセット
            elif key == ord("r"):
                self.annotations[frame_index] = []
                print(f"  [リセット] frame_index={frame_index} のアノテーションをリセットしました。")
                self._need_redraw = True

            # s : 保存
            elif key == ord("s"):
                if self._solve_and_save(frame_index):
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
def main():
    """メインエントリーポイント。コマンドライン引数を解析して GUI を起動する。"""
    parser = argparse.ArgumentParser(
        description="PnP Solver: イベントフレーム上でランドマークをアノテーションし、変換行列を求解する。"
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
        help="起動時に表示する RGB フレームインデックス (デフォルト: 最初のフレーム)",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"[エラー] config.json が見つかりません: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    config_dir = config_path.parent

    annotator = EventFrameAnnotator(config, config_dir)
    annotator.run(initial_frame_index=args.frame)


if __name__ == "__main__":
    main()
