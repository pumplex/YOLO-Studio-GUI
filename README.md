# 🚀 YOLO Training & Detection Studio

<p align="center">
  <img src="https://github.com/user-attachments/assets/e2a80191-b466-410e-997e-e390594bc786" alt="YOLO Studio Logo" width="120">
</p>

> A sleek, modern desktop GUI for training, evaluating, and deploying YOLO object-detection and segmentation models — no command line required.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c85f316b-cc44-44ba-8b10-017aaf2cd209" alt="Train tab – setup"></td>
    <td><img src="https://github.com/user-attachments/assets/b8165877-2c28-4545-8980-84770051f37e" alt="Train tab – advanced options"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/aacad0e3-6f69-4211-8fa3-3144ca6d7a26" alt="Train tab – live output & loss graph"></td>
    <td><img src="https://github.com/user-attachments/assets/297343b4-3e2a-435c-891b-4bd7039990fb" alt="Live Video tab – detection overlay"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/83e3db0f-475f-42de-b4d2-a261be429e9a" alt="Detect tab – image results"></td>
    <td><img src="https://github.com/user-attachments/assets/f1fba430-261d-495b-ad66-e165f2f37b8b" alt="Detect tab – video results"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/c34dc87f-d5ec-470d-a894-b092632b750d" alt="Benchmark tab – running"></td>
    <td><img src="https://github.com/user-attachments/assets/ef6a4dcb-d050-46f4-b999-54984dc49e02" alt="Benchmark tab – results table"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/0dd29480-a42e-416d-9d82-53d26731414b" alt="Benchmark tab – bar charts"></td>
    <td><img src="https://github.com/user-attachments/assets/b3706bdc-f0af-422e-b974-c55244d227bb" alt="Export tab"></td>
  </tr>
</table>

Whether you're training your first custom model or benchmarking multiple architectures side-by-side, **YOLO Studio** wraps the power of [Ultralytics](https://github.com/ultralytics/ultralytics) in a clean, tooltip-rich interface that gets out of your way. 🎯

---

## ✨ Features at a glance

| Feature | Details |
|---------|---------|
| 🏋 **Train** | Five task types — Detection, Segmentation, Classification, Pose Estimation, OBB Detection — across YOLOv8 → YOLOv12, with real-time log output, live loss graphs (box / cls / dfl), dual progress bars (epoch + inner batch/validation), and ETA display |
| 📈 **Loss graphs** | Live-updating box\_loss, cls\_loss, and dfl\_loss charts embedded in the Train tab; click any graph to open a larger popup |
| 🗂️ **Training queue** | Queue multiple training jobs, save/load job configs, and run them sequentially with **Run Queue** |
| 📦 **Roboflow ZIP import** | One-click extraction & auto-configuration of Roboflow datasets, with class names filled in automatically |
| 🔧 **Custom base model** | Fine-tune from any `.pt` weights file with an optional Resume toggle to continue an interrupted run |
| ⚙️ **Advanced training options** | Max training time, early-stop patience, checkpoint saving with configurable period, dataset caching, layer freezing, initial/final LR, momentum, weight decay, optimizer selection, validation toggle, and max detections |
| 🔍 **Detect** | Run inference on image / video folders with FP16, confidence threshold, worker count, and task-type override; browse results with a built-in viewer or open the full gallery |
| 📷 **Camera** | Live webcam detection with FP16 support and one-key frame capture |
| 🎬 **Live Video** | Seekable video playback (file **or** URL/stream) with real-time YOLO overlay, audio playback (via ffmpeg + sounddevice), volume slider, audio sync toggle, pause/resume, per-second jump buttons (−10 s / −5 s / +5 s / +10 s), screenshot (annotated or raw), FP16, confidence control, and task-type override for `.onnx` / `.engine` models |
| 📊 **Benchmark** | Compare any number of `.pt` models — evaluate on Test, Validation, Train, or a custom folder; colour-coded results table + interactive bar charts for mAP50, mAP50-95, precision, recall, speed, and size |
| ⬇ **Export** | ONNX · TensorRT Engine · CoreML · TF SavedModel · TFLite, with a live output log streamed in real time |
| 💡 **Tooltips** | Every control carries a contextual help tip |
| 🌙 **Appearance** | Dark / Light / System theme toggle |
| 🖥️ **Hardware** | Sidebar CPU ↔ GPU toggle; CUDA device name displayed when available |
| 💾 **Settings** | Option to store temporary files in RAM for faster processing |

---

## 🖥 Requirements

- Python **3.10 or later**
- Windows, macOS, or Linux
- (Optional but strongly recommended) NVIDIA GPU with CUDA 12.x for fast training

---

## ⚙️ Installation

### 1 — Clone this repository

```bash
git clone https://github.com/pumplex/YOLO-Studio-GUI.git
cd YOLO-Studio-GUI
```

### 2 — Create a virtual environment

**venv:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**Conda:**
```bash
conda create -n yolo-studio python=3.12
conda activate yolo-studio
```

### 3 — Install PyTorch

#### 🔥 GPU (NVIDIA CUDA 12.8 — recommended)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# For TensorRT export support:
pip install tensorrt
# For ONNX export support:
pip install onnx
```

#### 💻 CPU-only
```bash
pip install torch torchvision
```

> For other CUDA versions (11.8, 12.1 …) visit **https://pytorch.org/get-started/locally/**

### 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` includes `matplotlib` for benchmark charts.  If you skipped it, install manually: `pip install matplotlib`.

### 5 — Run

```bash
python main.py
```

---

## 📖 How to use

### 🏋 Train tab

Train your own YOLO model from scratch or fine-tune an existing one.

1. **Import Roboflow ZIP** *(optional)* — see the section below, or:
2. Select your **Training Data Folder** (see *Data format* below).
3. Choose a **Model Save Folder** for the output weights.
4. Pick a **Task Type**: Detection, Segmentation, Classification, Pose Estimation, or OBB Detection.
5. Select a **YOLO Model** from the dropdown (YOLOv8 → YOLOv12, all sizes), or load a **Custom Base Model** (`.pt`) and optionally enable **Resume** to continue an interrupted run.
6. Fill in **Image Size**, **Epochs**, **Batch Size**, **Workers**, and **Class Names**.
7. Expand **Advanced Training Options** for fine-grained control (see below).
8. Click **▶ Start Training** — live output streams to the log panel.

The **primary progress bar** tracks overall epoch progress with an ETA; a **secondary progress bar** shows inner-batch or validation progress per epoch. Three **live loss graphs** (box\_loss, cls\_loss, dfl\_loss) update as each epoch completes — click any graph to open a larger popup window.

ANSI escape codes and carriage-return rewrite sequences are automatically stripped so the log is always clean and readable.

> **Tip:** Hover over any control for a tooltip explaining what it does.

#### ⚙️ Advanced Training Options

| Option | What it does |
|--------|-------------|
| **Max Training Time** | Stop training after N hours (0 = no limit) |
| **Patience** | Early-stop after N epochs without improvement |
| **Save Checkpoints / Save Period** | Save a `.pt` every N epochs (−1 = last epoch only) |
| **Cache Dataset Images** | Pre-load images into RAM for faster training |
| **Freeze Layers** | Freeze the first N backbone layers (0 = disabled) |
| **Initial LR / Final LR** | Learning-rate schedule start and end values |
| **Momentum / Weight Decay** | SGD/Adam optimiser hyperparameters |
| **Optimizer** | auto · SGD · Adam · AdamW · NAdam · RAdam · RMSProp |
| **Validation** | Run validation pass each epoch (toggleable) |
| **Max Detections** | Maximum detections per image during validation |

#### 🗂️ Training queue

- Click **➕ Add to Queue** to enqueue the current configuration without starting immediately.
- Click **▶▶ Run Queue** to run all queued jobs sequentially.
- **Save Config / Load Config** — persist or restore the full training configuration to/from a JSON file.
- The queue panel lists pending jobs; individual jobs can be removed at any time.

#### 📦 Roboflow dataset import

Download your dataset from [roboflow.com](https://roboflow.com):
- **Export → Format: YOLOv8** → **Download ZIP**

Then in the Train tab:
1. Click **📦 Import Roboflow ZIP…**
2. Select the downloaded `.zip` file.
3. Choose where to extract it.
4. The app extracts the archive, patches paths, and **fills in class names automatically**.
5. Review the settings and click **▶ Start Training**.

Expected ZIP structure (handled automatically):
```
dataset.zip/
  data.yaml
  train/images/*.jpg    train/labels/*.txt
  valid/images/*.jpg    valid/labels/*.txt
  test/images/*.jpg     test/labels/*.txt  (optional)
```

#### 📁 Manual data format

If you're not using Roboflow, place image + annotation pairs in the same folder:

```
my_data/
  photo1.jpg   photo1.txt
  photo2.png   photo2.txt
  ...
```

Each `.txt` file uses standard YOLO format (one object per line):
```
<class_id> <x_center> <y_center> <width> <height>
```
All coordinates are normalised 0 – 1 relative to image dimensions. The app automatically splits your data **80 % train / 20 % val**.

---

### 🔍 Detect tab

Run YOLO inference on a folder of images or videos.

1. Click **Select Images/Videos Folder**.
2. Click **Select Model (.pt / .onnx / .engine)** to load your trained weights.
3. Optionally adjust **Confidence Threshold**, enable **FP16** (GPU only), set **Data Workers**, and choose a **Task** override (required for exported ONNX/TensorRT models).
4. Click **▶ Start Detection**.
5. Results are displayed in the viewer — use **◀ ▶** to browse or **🖼 Open Gallery** to see all results at once.

---

### 📷 Camera tab

Real-time detection on a live webcam feed.

1. Select a **Model (.pt)** and optionally a **Save Folder**.
2. Enter the **Camera ID** (usually `0` for the built-in webcam).
3. Enable **FP16** for faster GPU inference (optional).
4. Click **▶ START** — detection begins immediately.
5. Press **Enter** at any time to capture and save the current frame.
6. Click **■ STOP** to end the session.

---

### 🎬 Live Video tab

Play back a video file **or** live stream/URL with real-time YOLO detection overlaid.

| Control | What it does |
|---------|-------------|
| **📂 Video** | Pick any video file (`.mp4`, `.avi`, `.mov`, `.mkv`, …) |
| **🔗 URL** | Enter an RTSP/HTTP stream URL instead of a local file |
| **🤖 Model** | Load a `.pt`, `.onnx`, or TensorRT `.engine` model |
| **Task** | Override the model task type (required for ONNX/TensorRT models without embedded metadata) |
| **Seek slider** | Drag to jump to any point in the video |
| **−10s / −5s / +5s / +10s** | Jump backwards or forwards by a fixed amount |
| **⏸ Pause / ▶ Resume** | Freeze and continue playback without restarting |
| **📷 Screenshot** | Save the current frame as PNG — choose *With Detections* (annotated) or *Raw Frame* (original) |
| **FP16** | Enable half-precision inference for ~2× speed on NVIDIA GPUs |
| **Conf** | Minimum detection confidence shown in the overlay |
| **🔊 Audio** | Toggle audio playback (extracted via ffmpeg, streamed with sounddevice) |
| **Sync** | Synchronise audio speed to actual video playback rate |
| **Vol** | Playback volume slider (0 – 100 %) |
| **▶ PLAY / ■ STOP** | Start or stop video playback |

The status bar shows the current filename, frame position, detection count, and live FPS.

**Performance tips:**
- Enable **FP16** on an NVIDIA GPU for the biggest speed boost.
- Use a pre-exported **ONNX** or **TensorRT** model instead of `.pt` for faster inference.
- Reduce the **Conf** threshold slightly to avoid spending time on borderline detections.

---

### 📊 Benchmark tab

Compare multiple trained models on the same dataset to find the best one for your use case.

1. Click **➕ Add Model(s)** to load one or more `.pt` files.
2. Under **Dataset Source**, click **Browse YAML…** to select a `data.yaml`, or **Browse Folder…** to point directly at an image folder.
3. Set the **Image Size** (match training size).
4. Choose which split to **Evaluate On**: Test set · Valid set · Train set · Folder set.
5. Click **▶ Run Benchmark** — a live log on the right panel streams results as each model is evaluated.

Results are displayed in a colour-coded table:

| Column | What it means |
|--------|---------------|
| **Accuracy (mAP50 ↑)** | How often the model detects the right thing, at least 50 % overlap — higher is better |
| **Fine Accuracy (mAP50-95 ↑)** | Stricter accuracy across multiple overlap thresholds — harder to score well on |
| **Precision ↑** | Of all detections made, what fraction were correct? (fewer false alarms) |
| **Recall ↑** | Of all real objects in images, what fraction were found? (fewer misses) |
| **Speed (ms/img ↓)** | Inference time per image — lower is faster |
| **Size (MB)** | Model file size on disk |

🟢 **Green** = most accurate · 🔵 **Blue** = fastest · 🟣 **Purple** = lightest

After results appear, click **📊 Show Bar Charts** to open an interactive chart window comparing all metrics side-by-side. The chart window includes a standard matplotlib toolbar for zooming, panning, and saving the chart as an image.

---

### ⬇ Export tab

Convert a trained model to a deployment format.

1. Click **Browse .pt…** to load your trained weights.
2. Select an **Export Format**:
   - **ONNX** — universal, runs anywhere (CPU · GPU · accelerators)
   - **TensorRT Engine** — maximum throughput on NVIDIA GPUs *(see note below)*
   - **CoreML** — Apple silicon / macOS / iOS
   - **TF SavedModel** / **TFLite** — TensorFlow / mobile / embedded
3. Click **⬇ Export Model**.

The right panel shows a **live log** of the export process — all Ultralytics output is streamed in real time so you can see exactly what is happening.

> ℹ️ **TensorRT note:** The exported `.engine` file is compiled for the exact GPU it was built on
> and cannot be transferred to a different GPU model. TensorRT ≥ 8 and CUDA must be installed
> (`pip install tensorrt`). This is an **inference optimisation**, not a training format.

---

## 🏗️ Project structure

```
YOLO-Studio-GUI/
├── main.py               # GUI entry point (all tabs, sidebar, training queue, live graphs)
├── requirements.txt
├── README.md
└── src/
    ├── train.py          # Training logic + YAML creation + data preparation
    ├── detect.py         # Image / video batch detection
    ├── camera.py         # Live camera detection
    ├── dataset.py        # Roboflow ZIP import utilities
    ├── calculate_metrics.py  # Benchmark metrics helpers
    └── xml_to_txt.py     # Pascal VOC XML → YOLO TXT annotation converter
```

---

## 🤝 Credits & licence

Built on top of [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter).  
Original project by [SpreadKnowledge](https://github.com/SpreadKnowledge/YOLO_train_detection_GUI).  
Modernised and extended for [pumplex/YOLO-Studio-GUI](https://github.com/pumplex/YOLO-Studio-GUI).

Licensed under the terms included in [LICENSE](LICENSE).
