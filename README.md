# PnP Calibration Tool

RGB カメラ（MediaPipe 3D 顔メッシュ）とイベントカメラ間の **外部パラメータ（変換行列）** を  
PnP（Perspective-n-Point）問題として導出するシステムです。

---

## ディレクトリ構成

```
pnp_calibration/
├── config.json               # 設定ファイル（パス・パラメータ）
├── requirements.txt          # Python 依存パッケージ
├── event_loader.py           # イベントデータ読み込み共通モジュール
├── main_pnp_solver.py        # メインツール（アノテーション + PnP 求解）
├── verify_reprojection.py    # 検証ツール（再投影確認）
├── generate_dummy_data.py    # テスト用ダミーデータ生成スクリプト
├── input/                    # 入力ファイル置き場
│   ├── sync_log.csv
│   ├── sync_params.json
│   ├── events.csv
│   ├── landmark.csv
│   └── calibration.json
└── output/                   # 出力ファイル置き場
    ├── transform_matrix.json
    └── annotated_points.json
```

---

## セットアップ

```bash
# 1. 仮想環境の作成
python3 -m venv .venv
source .venv/bin/activate

# 2. 依存パッケージのインストール
pip install -r requirements.txt
```

---

## 入力ファイルの仕様

| ファイル | 説明 |
|---|---|
| `sync_log.csv` | RGB 動画フレームのタイムスタンプ（`frame_index`, `timestamp_ms`, `led_status`） |
| `sync_params.json` | 時間同期パラメータ `A`, `B`（`t_rgb [ms] = A × t_event [µs] + B`） |
| `events.csv` | イベントカメラデータ（`x`, `y`, `polarity`, `timestamp [µs]`） |
| `landmark.csv` | MediaPipe 3D 顔メッシュ（`frame_index`, `face_index`, `landmark_index`, `x_norm`, `y_norm`, `z_norm`） |
| `calibration.json` | イベントカメラ内部パラメータ（`intrinsics` 3×3, `distortion` 5要素） |

---

## 使い方

### Step 0: ダミーデータで動作確認（オプション）

```bash
python generate_dummy_data.py
```

`input/` に全てのダミーファイルが生成されます。

### Step 1: メインツール（アノテーション & PnP 求解）

```bash
python main_pnp_solver.py --config config.json --frame 0
```

**GUI 操作:**

| キー | 操作 |
|---|---|
| `n` / `→` | 次のフレームへ |
| `p` / `←` | 前のフレームへ |
| `u` | 直前のクリックを Undo |
| `r` | 現フレームのアノテーションをリセット |
| `s` | PnP を計算して `output/` に保存 |
| `q` / `Esc` | 終了 |

アノテーション手順:
1. イベントフレームが表示される
2. 上部に「次にクリックすべき部位（例: 右目の目頭）」が表示される
3. 対応する点をマウスでクリック（全 7 点）
4. `s` キーで PnP 求解 & 保存

### Step 2: 検証ツール（再投影確認）

```bash
python verify_reprojection.py --config config.json
```

- `output/transform_matrix.json` の `rvec` / `tvec` を読み込み
- 全顔メッシュ点（最大 478 点）をイベントフレームに赤い点で投影
- アノテーション済み点は緑の円で表示

---

## 出力ファイルの仕様

### `output/transform_matrix.json`

```json
{
  "frame_index": 0,
  "rvec": [rx, ry, rz],
  "tvec": [tx, ty, tz],
  "reprojection_errors_px": [...],
  "mean_reprojection_error_px": 1.23
}
```

### `output/annotated_points.json`

アノテーションした 7 点の 2D/3D 座標ペアと再投影結果:

```json
{
  "frame_index": 0,
  "landmarks": [
    {
      "id": 308,
      "name": "Left of mouth",
      "point_2d": [u, v],
      "point_3d": [x, y, z],
      "reprojected_2d": [u', v'],
      "error_px": 0.52
    },
    ...
  ]
}
```

---

## config.json パラメータ説明

| キー | 説明 |
|---|---|
| `paths.*` | 各入出力ファイルのパス（config.json からの相対パス） |
| `event_frame.width` | イベントフレームの幅 [pixels] |
| `event_frame.height` | イベントフレームの高さ [pixels] |
| `event_frame.integration_time_ms` | イベント積分時間 [ms]（この幅でイベントを収集） |
| `target_landmarks[].id` | MediaPipe ランドマーク ID |
| `target_landmarks[].name` | 部位名（GUI でのガイド表示に使用） |

---

## 注意事項

- `events.csv` が巨大な場合でも、バイナリサーチ（`np.searchsorted`）による部分読み込みで高速・省メモリに処理します。
- 再投影誤差が大きい場合は、アノテーション精度を見直してください（`r` キーでリセット可能）。
- `cv2.solvePnP` は `SOLVEPNP_ITERATIVE` フラグを使用しています。点数が多い場合は `SOLVEPNP_EPNP` も試してみてください。
