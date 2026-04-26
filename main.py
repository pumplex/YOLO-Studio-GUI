# version 1.0.0 – modernised UI, bug-fixed, segmentation models, custom model loader, TensorRT export

import os
import sys
import cv2
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import subprocess
import mimetypes
from pathlib import Path
from queue import Queue, Empty
from src.train import create_yaml
from src.detect import detect_images, is_valid_image
from src.camera import CameraDetection

mimetypes.init()


# ─────────────────────────────────────────────────────────────────────────────
#  Tooltip helper
# ─────────────────────────────────────────────────────────────────────────────
class Tooltip:
    """Show a brief help tip when the user hovers over any tk/ctk widget."""

    def __init__(self, widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        try:
            x = self.widget.winfo_rootx() + 24
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffc0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 10),
            wraplength=340,
            padx=7,
            pady=5,
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._tip:
            self._tip.destroy()
            self._tip = None


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────
def get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor.  Cross-platform."""
    try:
        if sys.platform == "win32":
            import ctypes as _c
            u = _c.windll.user32
            return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        pass
    try:
        import tkinter as _tk
        _r = _tk.Tk()
        _r.withdraw()
        w, h = _r.winfo_screenwidth(), _r.winfo_screenheight()
        _r.destroy()
        return w, h
    except Exception:
        return 1280, 800


def normalize_path(path: str) -> str:
    if not path:
        return path
    return str(Path(path).resolve())


def clear_frame(frame) -> None:
    for widget in frame.winfo_children():
        widget.destroy()


def _safe_label_configure(label, **kwargs) -> None:
    """Update a label widget only if it still exists."""
    try:
        if label is not None and label.winfo_exists():
            label.configure(**kwargs)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Model catalogue
# ─────────────────────────────────────────────────────────────────────────────
DETECTION_MODELS: list[str] = [
    "YOLOv8-Nano",      "YOLOv8-Small",     "YOLOv8-Medium",
    "YOLOv8-Large",     "YOLOv8-ExtraLarge",
    "YOLOv9-Compact",   "YOLOv9-Enhanced",
    "YOLOv10-Nano",     "YOLOv10-Small",    "YOLOv10-Medium",
    "YOLOv10-Balanced", "YOLOv10-Large",    "YOLOv10-ExtraLarge",
    "YOLOv11-Nano",     "YOLOv11-Small",    "YOLOv11-Medium",
    "YOLOv11-Large",    "YOLOv11-ExtraLarge",
    "YOLOv12-Nano",     "YOLOv12-Small",    "YOLOv12-Medium",
    "YOLOv12-Large",    "YOLOv12-ExtraLarge",
]

SEGMENTATION_MODELS: list[str] = [
    "YOLOv8-Nano-Seg",       "YOLOv8-Small-Seg",      "YOLOv8-Medium-Seg",
    "YOLOv8-Large-Seg",      "YOLOv8-ExtraLarge-Seg",
    "YOLOv11-Nano-Seg",      "YOLOv11-Small-Seg",     "YOLOv11-Medium-Seg",
    "YOLOv11-Large-Seg",     "YOLOv11-ExtraLarge-Seg",
]

MODEL_MAP: dict[str, str] = {
    # Detection
    "YOLOv8-Nano":       "yolov8n",   "YOLOv8-Small":       "yolov8s",
    "YOLOv8-Medium":     "yolov8m",   "YOLOv8-Large":       "yolov8l",
    "YOLOv8-ExtraLarge": "yolov8x",
    "YOLOv9-Compact":    "yolov9c",   "YOLOv9-Enhanced":    "yolov9e",
    "YOLOv10-Nano":      "yolov10n",  "YOLOv10-Small":      "yolov10s",
    "YOLOv10-Medium":    "yolov10m",  "YOLOv10-Balanced":   "yolov10b",
    "YOLOv10-Large":     "yolov10l",  "YOLOv10-ExtraLarge": "yolov10x",
    "YOLOv11-Nano":      "yolo11n",   "YOLOv11-Small":      "yolo11s",
    "YOLOv11-Medium":    "yolo11m",   "YOLOv11-Large":      "yolo11l",
    "YOLOv11-ExtraLarge":"yolo11x",
    "YOLOv12-Nano":      "yolo12n",   "YOLOv12-Small":      "yolo12s",
    "YOLOv12-Medium":    "yolo12m",   "YOLOv12-Large":      "yolo12l",
    "YOLOv12-ExtraLarge":"yolo12x",
    # Segmentation
    "YOLOv8-Nano-Seg":        "yolov8n-seg",  "YOLOv8-Small-Seg":      "yolov8s-seg",
    "YOLOv8-Medium-Seg":      "yolov8m-seg",  "YOLOv8-Large-Seg":      "yolov8l-seg",
    "YOLOv8-ExtraLarge-Seg":  "yolov8x-seg",
    "YOLOv11-Nano-Seg":       "yolo11n-seg",  "YOLOv11-Small-Seg":     "yolo11s-seg",
    "YOLOv11-Medium-Seg":     "yolo11m-seg",  "YOLOv11-Large-Seg":     "yolo11l-seg",
    "YOLOv11-ExtraLarge-Seg": "yolo11x-seg",
}

EXPORT_FORMATS: list[str] = ["ONNX", "TensorRT Engine", "CoreML", "TF SavedModel", "TFLite"]
EXPORT_FORMAT_MAP: dict[str, str] = {
    "ONNX":          "onnx",
    "TensorRT Engine": "engine",
    "CoreML":        "coreml",
    "TF SavedModel": "saved_model",
    "TFLite":        "tflite",
}


# ─────────────────────────────────────────────────────────────────────────────
#  Global application state
# ─────────────────────────────────────────────────────────────────────────────
project_name:              str = ""
train_data_path:           str = ""
model_save_path:           str = ""
custom_model_path:         str = ""   # optional .pt for training base
input_size:                str = ""
epochs:                    str = ""
batch_size:                str = ""
class_names:               list[str] = []
image_paths:               list[str] = []
current_image_index:       int = 0
detection_model_path:      str = ""
detection_images_folder_path: str = ""
detection_save_dir:        str = ""
export_model_path:         str = ""
camera_detection:          CameraDetection | None = None

# Widget references populated inside show_* functions
output_textbox         = None
progress_bar           = None
detection_progress_bar = None
image_label            = None
image_index_label      = None
selected_model_var     = None   # StringVar for model dropdown
task_type_var          = None   # StringVar "Detection" / "Segmentation"
model_menu_widget      = None   # CTkOptionMenu reference
_camera_bar            = None   # bottom bar in camera view (holds start/stop btn)

# Status labels (set inside each show_* function, None when not visible)
train_data_label  = None
model_save_label  = None
custom_model_label = None
detect_folder_label = None
detect_model_label  = None
export_model_label  = None
export_status_label = None

output_queue: Queue = Queue()


# ─────────────────────────────────────────────────────────────────────────────
#  Output-queue consumer  (runs on main thread via root.after)
# ─────────────────────────────────────────────────────────────────────────────
def update_output_textbox() -> None:
    global output_textbox
    try:
        if output_textbox is not None and output_textbox.winfo_exists():
            line = output_queue.get_nowait()
            output_textbox.insert("end", line)
            output_textbox.yview_moveto(1)
    except Empty:
        pass
    except Exception:
        pass
    finally:
        root.after(100, update_output_textbox)


# ─────────────────────────────────────────────────────────────────────────────
#  Image navigation (detection results)
# ─────────────────────────────────────────────────────────────────────────────
def update_image() -> None:
    global current_image_index, image_label, image_paths, image_index_label
    if not image_paths or image_label is None:
        return
    try:
        if not image_label.winfo_exists():
            return
    except Exception:
        return

    _safe_label_configure(image_index_label, text=f"{current_image_index + 1}/{len(image_paths)}")
    img = Image.open(image_paths[current_image_index])
    img_w, img_h = img.size
    max_w = max(1, image_label.winfo_width())
    max_h = max(1, image_label.winfo_height())
    scale = min(max_w / img_w, max_h / img_h)
    img = img.resize(
        (max(1, int(img_w * scale)), max(1, int(img_h * scale))),
        Image.Resampling.LANCZOS,
    )
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo  # keep reference alive


def show_next_image() -> None:
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index + 1) % len(image_paths)
        update_image()


def show_prev_image() -> None:
    global current_image_index, image_paths
    if image_paths:
        current_image_index = (current_image_index - 1) % len(image_paths)
        update_image()


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
def on_sidebar_select(key: str) -> None:
    # Reset all shared label references before building a new view
    global train_data_label, model_save_label, custom_model_label
    global detect_folder_label, detect_model_label
    global export_model_label, export_status_label
    global output_textbox, progress_bar, detection_progress_bar
    global image_label, image_index_label, model_menu_widget
    global selected_model_var, task_type_var, _camera_bar

    clear_frame(main_frame)

    train_data_label = model_save_label = custom_model_label = None
    detect_folder_label = detect_model_label = None
    export_model_label = export_status_label = None
    image_label = image_index_label = None
    model_menu_widget = None
    _camera_bar = None

    if key == "Train":
        show_ai_train_window()
    elif key == "Detect":
        show_image_detection_window()
    elif key == "Camera":
        show_camera_detection_window()
    elif key == "Export":
        show_export_window()


# ─────────────────────────────────────────────────────────────────────────────
#  Train window
# ─────────────────────────────────────────────────────────────────────────────
def _on_task_type_change(*_args) -> None:
    """Repopulate the model dropdown when the task type changes."""
    global model_menu_widget, selected_model_var, task_type_var
    if task_type_var is None or model_menu_widget is None:
        return
    options = DETECTION_MODELS if task_type_var.get() == "Detection" else SEGMENTATION_MODELS
    selected_model_var.set(options[0])
    model_menu_widget.configure(values=options)


def show_ai_train_window() -> None:
    global output_textbox, progress_bar, selected_model_var, task_type_var, model_menu_widget
    global train_data_label, model_save_label, custom_model_label

    # ── Left: scrollable configuration panel ─────────────────────────────
    config_panel = ctk.CTkScrollableFrame(
        master=main_frame,
        label_text="Training Configuration",
        label_font=("Segoe UI", 14, "bold"),
        corner_radius=8,
    )
    config_panel.place(relx=0, rely=0, relwidth=0.41, relheight=1.0)

    # ── Right: log / output panel ─────────────────────────────────────────
    log_panel = ctk.CTkFrame(master=main_frame, corner_radius=8)
    log_panel.place(relx=0.42, rely=0, relwidth=0.58, relheight=1.0)

    PAD  = {"padx": 14, "pady": 5}
    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)
    FENT = ("Segoe UI", 13)

    def _lbl(text: str):
        l = ctk.CTkLabel(config_panel, text=text, font=FLAB, anchor="w")
        l.pack(fill="x", padx=14, pady=(8, 1))
        return l

    def _sep():
        ctk.CTkFrame(config_panel, height=1, fg_color="gray50").pack(
            fill="x", padx=14, pady=4
        )

    # ── Project name ──────────────────────────────────────────────────────
    _lbl("Project Name")
    project_name_entry = ctk.CTkEntry(
        config_panel, placeholder_text="e.g.  my_detector", font=FENT, height=36
    )
    project_name_entry.pack(fill="x", **PAD)
    Tooltip(
        project_name_entry,
        "A short alphanumeric name for this run.\n"
        "Training results and the YAML config are saved under this name.",
    )
    _sep()

    # ── Training data folder ──────────────────────────────────────────────
    _lbl("Training Data Folder")
    train_data_btn = ctk.CTkButton(
        config_panel, text="Browse…", font=FBTN, height=36, command=select_train_data
    )
    train_data_btn.pack(fill="x", **PAD)
    Tooltip(
        train_data_btn,
        "Select a folder that contains paired image files and YOLO-format\n"
        "annotation .txt files with the same base name.\n\n"
        "Expected layout:\n"
        "  folder/\n"
        "    photo1.jpg   photo1.txt\n"
        "    photo2.png   photo2.txt  …\n\n"
        "The app will automatically split 80 % → train, 20 % → val.",
    )
    train_data_label = ctk.CTkLabel(
        config_panel, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    train_data_label.pack(fill="x", padx=14)
    _sep()

    # ── Save folder ───────────────────────────────────────────────────────
    _lbl("Model Save Folder")
    model_save_btn = ctk.CTkButton(
        config_panel, text="Browse…", font=FBTN, height=36, command=select_model_save_folder
    )
    model_save_btn.pack(fill="x", **PAD)
    Tooltip(model_save_btn, "Choose where the trained model weights and results will be saved.")
    model_save_label = ctk.CTkLabel(
        config_panel, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    model_save_label.pack(fill="x", padx=14)
    _sep()

    # ── Task type ─────────────────────────────────────────────────────────
    _lbl("Task Type")
    task_type_var = ctk.StringVar(value="Detection")
    task_frame = ctk.CTkFrame(config_panel, fg_color="transparent")
    task_frame.pack(fill="x", **PAD)
    det_radio = ctk.CTkRadioButton(
        task_frame, text="Detection",
        variable=task_type_var, value="Detection",
        command=_on_task_type_change, font=FLAB,
    )
    det_radio.pack(side="left", padx=(0, 24))
    seg_radio = ctk.CTkRadioButton(
        task_frame, text="Segmentation",
        variable=task_type_var, value="Segmentation",
        command=_on_task_type_change, font=FLAB,
    )
    seg_radio.pack(side="left")
    Tooltip(
        task_frame,
        "Detection  – predicts bounding boxes around objects.\n"
        "Segmentation – predicts pixel-level instance masks.\n\n"
        "Segmentation requires seg-compatible model weights and polygon\n"
        "annotations in your dataset.",
    )

    # ── YOLO model dropdown ───────────────────────────────────────────────
    _lbl("YOLO Model")
    selected_model_var = ctk.StringVar(value=DETECTION_MODELS[0])
    model_menu_widget = ctk.CTkOptionMenu(
        config_panel,
        variable=selected_model_var,
        values=DETECTION_MODELS,
        font=FBTN,
        dropdown_font=FBTN,
        height=36,
    )
    model_menu_widget.pack(fill="x", **PAD)
    Tooltip(
        model_menu_widget,
        "Pre-trained Ultralytics weights used as the training starting point.\n\n"
        "Nano / Small  – fastest, least accurate; ideal for edge devices.\n"
        "Medium        – balanced speed and accuracy.\n"
        "Large / ExtraLarge – most accurate; needs more GPU memory.\n\n"
        "Segmentation variants require seg-format polygon annotations.",
    )
    _sep()

    # ── Custom base model (optional) ──────────────────────────────────────
    _lbl("Custom Base Model  (optional)")
    custom_model_btn = ctk.CTkButton(
        config_panel, text="Browse .pt…", font=FBTN, height=36, command=select_custom_model
    )
    custom_model_btn.pack(fill="x", **PAD)
    Tooltip(
        custom_model_btn,
        "Load your own .pt file as the training starting point.\n"
        "When set, this overrides the YOLO Model dropdown above.\n\n"
        "Useful for fine-tuning an already-trained custom model.",
    )
    custom_model_label = ctk.CTkLabel(
        config_panel, text="Using built-in pretrained weights",
        font=("Segoe UI", 11), text_color="gray", anchor="w",
    )
    custom_model_label.pack(fill="x", padx=14)
    clear_custom_btn = ctk.CTkButton(
        config_panel, text="Clear custom model", font=("Segoe UI", 11),
        height=28, fg_color="gray50", hover_color="gray35",
        command=clear_custom_model,
    )
    clear_custom_btn.pack(fill="x", padx=14, pady=(2, 4))
    _sep()

    # ── Numeric training params ───────────────────────────────────────────
    _lbl("Image Size  (e.g. 640)")
    input_size_entry = ctk.CTkEntry(
        config_panel, placeholder_text="640", font=FENT, height=36
    )
    input_size_entry.pack(fill="x", **PAD)
    Tooltip(
        input_size_entry,
        "Square resolution fed into the network (pixels).\n"
        "640 is standard.  Use 416 for faster training or\n"
        "1280 for higher precision on large images.",
    )

    _lbl("Epochs  (e.g. 100)")
    epochs_entry = ctk.CTkEntry(
        config_panel, placeholder_text="100", font=FENT, height=36
    )
    epochs_entry.pack(fill="x", **PAD)
    Tooltip(
        epochs_entry,
        "Number of full passes over the training dataset.\n"
        "More epochs → longer training, potentially better accuracy.\n"
        "Start with 50–100; increase if validation loss is still improving.",
    )

    _lbl("Batch Size  (e.g. 16)")
    batch_size_entry = ctk.CTkEntry(
        config_panel, placeholder_text="16", font=FENT, height=36
    )
    batch_size_entry.pack(fill="x", **PAD)
    Tooltip(
        batch_size_entry,
        "Images processed per gradient-update step.\n"
        "Reduce (e.g. 8 or 4) if you run out of GPU/CPU memory.\n"
        "Larger batches generally train faster but need more RAM.",
    )
    _sep()

    # ── Class names ───────────────────────────────────────────────────────
    _lbl("Class Names  (one per line)")
    class_names_text = ctk.CTkTextbox(config_panel, font=FENT, height=110)
    class_names_text.pack(fill="x", **PAD)
    Tooltip(
        class_names_text,
        "Enter each object class on its own line, in the same order as the\n"
        "class IDs used in your annotation .txt files.\n\n"
        "Example:\n"
        "  cat\n"
        "  dog\n"
        "  car",
    )
    _sep()

    # ── Start Training button ─────────────────────────────────────────────
    start_btn = ctk.CTkButton(
        config_panel,
        text="▶  Start Training",
        command=lambda: start_training(
            project_name_entry, input_size_entry, epochs_entry,
            batch_size_entry, class_names_text,
        ),
        fg_color="#2e7d32",
        hover_color="#1b5e20",
        font=("Segoe UI", 15, "bold"),
        height=50,
        text_color="white",
        corner_radius=8,
    )
    start_btn.pack(fill="x", padx=14, pady=12)

    # ── Log panel ─────────────────────────────────────────────────────────
    ctk.CTkLabel(
        log_panel, text="Training Output", font=("Segoe UI", 14, "bold")
    ).pack(anchor="w", padx=12, pady=(10, 4))

    output_textbox = ctk.CTkTextbox(
        log_panel, font=("Courier New", 12), corner_radius=8
    )
    output_textbox.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    progress_bar = ctk.CTkProgressBar(
        log_panel, progress_color="#43a047", mode="indeterminate", indeterminate_speed=0.7
    )
    progress_bar.pack(fill="x", padx=12, pady=(0, 10))


# ─────────────────────────────────────────────────────────────────────────────
#  Image / Video Detection window
# ─────────────────────────────────────────────────────────────────────────────
def show_image_detection_window() -> None:
    global image_label, detection_progress_bar, image_index_label
    global detect_folder_label, detect_model_label

    # Image display
    image_label = tk.Label(main_frame, bg="#111827")
    image_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.86)

    # Bottom control bar
    bar = ctk.CTkFrame(main_frame, corner_radius=0, height=80)
    bar.place(relx=0, rely=0.87, relwidth=1.0, relheight=0.13)

    FONT = ("Segoe UI", 12)

    # Status labels (row 1)
    detect_folder_label = ctk.CTkLabel(
        bar, text="No folder selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    detect_folder_label.place(relx=0.01, rely=0.03, relwidth=0.46, relheight=0.38)

    detect_model_label = ctk.CTkLabel(
        bar, text="No model selected", font=("Segoe UI", 11),
        text_color="gray", anchor="w",
    )
    detect_model_label.place(relx=0.50, rely=0.03, relwidth=0.48, relheight=0.38)

    # Buttons (row 2)
    sel_folder_btn = ctk.CTkButton(
        bar, text="Select Images/Videos Folder",
        command=select_detection_images_folder, font=FONT, height=34,
    )
    sel_folder_btn.place(relx=0.01, rely=0.48, relwidth=0.21, relheight=0.46)
    Tooltip(sel_folder_btn, "Pick a folder with images or videos to run YOLO detection on.")

    sel_model_btn = ctk.CTkButton(
        bar, text="Select Model (.pt)",
        command=select_detection_model, font=FONT, height=34,
    )
    sel_model_btn.place(relx=0.24, rely=0.48, relwidth=0.15, relheight=0.46)
    Tooltip(sel_model_btn, "Choose a trained YOLO .pt weights file for inference.")

    start_det_btn = ctk.CTkButton(
        bar, text="▶  Start Detection",
        command=lambda: [detection_progress_bar.start(), start_image_detection()],
        fg_color="#1565c0", hover_color="#0d47a1",
        font=("Segoe UI", 14, "bold"), height=34, text_color="white",
    )
    start_det_btn.place(relx=0.41, rely=0.48, relwidth=0.18, relheight=0.46)

    prev_btn = ctk.CTkButton(
        bar, text="◀", command=show_prev_image,
        fg_color="#1976d2", font=("Segoe UI", 20, "bold"), height=34,
    )
    prev_btn.place(relx=0.64, rely=0.48, relwidth=0.07, relheight=0.46)

    next_btn = ctk.CTkButton(
        bar, text="▶", command=show_next_image,
        fg_color="#1976d2", font=("Segoe UI", 20, "bold"), height=34,
    )
    next_btn.place(relx=0.72, rely=0.48, relwidth=0.07, relheight=0.46)

    image_index_label = ctk.CTkLabel(bar, text="", font=("Segoe UI", 14))
    image_index_label.place(relx=0.80, rely=0.48, relwidth=0.09, relheight=0.46)

    detection_progress_bar = ctk.CTkProgressBar(
        bar, progress_color="#43a047", mode="indeterminate"
    )
    detection_progress_bar.place(relx=0.01, rely=0.96, relwidth=0.97, relheight=0.03)


# ─────────────────────────────────────────────────────────────────────────────
#  Camera Detection window
# ─────────────────────────────────────────────────────────────────────────────
def show_camera_detection_window() -> None:
    global camera_detection, camera_id_entry, image_label, _camera_bar

    camera_detection = None

    # Camera stream display
    image_label = tk.Label(main_frame, bg="#0d0d0d")
    image_label.place(relx=0, rely=0, relwidth=1.0, relheight=0.93)

    # Bottom control bar
    bar = ctk.CTkFrame(main_frame, corner_radius=0, height=50)
    bar.place(relx=0, rely=0.93, relwidth=1.0, relheight=0.07)
    _camera_bar = bar

    FONT = ("Segoe UI", 12)

    sel_model_btn = ctk.CTkButton(
        bar, text="Select Model (.pt)",
        command=select_detection_model, font=FONT, height=34,
    )
    sel_model_btn.place(relx=0.01, rely=0.1, relwidth=0.14, relheight=0.8)
    Tooltip(sel_model_btn, "Choose the YOLO .pt model file used for live camera inference.")

    sel_save_btn = ctk.CTkButton(
        bar, text="Save Folder",
        command=select_camera_save_folder, font=FONT, height=34,
    )
    sel_save_btn.place(relx=0.17, rely=0.1, relwidth=0.11, relheight=0.8)
    Tooltip(sel_save_btn, "Folder where captured frames are saved when you press Enter.")

    camera_id_entry = ctk.CTkEntry(
        bar, placeholder_text="Camera ID  (e.g. 0)", font=FONT, height=34
    )
    camera_id_entry.place(relx=0.30, rely=0.1, relwidth=0.15, relheight=0.8)
    Tooltip(
        camera_id_entry,
        "Index of the camera to open.\n"
        "0 = default webcam, 1 = second camera, etc.\n"
        "On Linux you may need to use /dev/video0 style paths.",
    )

    hint = ctk.CTkLabel(
        bar,
        text="Press  Enter  to capture & save a frame",
        font=("Segoe UI", 11),
        text_color="gray",
    )
    hint.place(relx=0.48, rely=0.1, relwidth=0.28, relheight=0.8)

    start_cam_btn = ctk.CTkButton(
        bar, text="▶  START",
        command=start_camera_detection,
        fg_color="#2e7d32", hover_color="#1b5e20",
        font=("Segoe UI", 14, "bold"), height=34, text_color="white",
    )
    start_cam_btn.place(relx=0.80, rely=0.1, relwidth=0.18, relheight=0.8)
    bar._start_btn = start_cam_btn   # stash ref for start/stop toggle

    root.bind("<Return>", lambda _e: save_callback())
    image_label.update_idletasks()


# ─────────────────────────────────────────────────────────────────────────────
#  Export window
# ─────────────────────────────────────────────────────────────────────────────
def show_export_window() -> None:
    global export_model_label, export_status_label, export_model_path
    export_model_path = ""

    FLAB = ("Segoe UI", 13)
    FBTN = ("Segoe UI", 13)

    ctk.CTkLabel(
        main_frame, text="Export Trained Model",
        font=("Segoe UI", 20, "bold"),
    ).place(relx=0.5, rely=0.06, anchor="center")

    # Model file selection
    ctk.CTkLabel(main_frame, text="Trained model (.pt)", font=FLAB).place(
        relx=0.25, rely=0.14, anchor="center"
    )
    sel_btn = ctk.CTkButton(
        main_frame, text="Browse .pt…", font=FBTN, height=38,
        command=select_export_model,
    )
    sel_btn.place(relx=0.25, rely=0.21, anchor="center", relwidth=0.30)
    Tooltip(sel_btn, "Select the trained YOLO .pt model you want to export.")

    export_model_label = ctk.CTkLabel(
        main_frame, text="No model selected",
        font=("Segoe UI", 11), text_color="gray",
    )
    export_model_label.place(relx=0.25, rely=0.28, anchor="center", relwidth=0.42)

    # Format selector
    ctk.CTkLabel(main_frame, text="Export Format", font=FLAB).place(
        relx=0.25, rely=0.36, anchor="center"
    )
    export_fmt_var = ctk.StringVar(value=EXPORT_FORMATS[0])
    fmt_menu = ctk.CTkOptionMenu(
        main_frame, variable=export_fmt_var, values=EXPORT_FORMATS,
        font=FBTN, height=38,
    )
    fmt_menu.place(relx=0.25, rely=0.43, anchor="center", relwidth=0.30)
    Tooltip(
        fmt_menu,
        "ONNX          – universal format; runs on CPU, GPU, or dedicated accelerators.\n"
        "TensorRT Engine – maximum throughput on NVIDIA GPUs; device-specific.\n"
        "CoreML        – Apple devices (macOS / iOS).\n"
        "TF SavedModel – TensorFlow ecosystem.\n"
        "TFLite        – mobile / embedded TensorFlow.",
    )

    # TensorRT information panel
    _trt_note = (
        "ℹ️  TensorRT notes\n\n"
        "Exporting to a TensorRT .engine file compiles the model into GPU-specific\n"
        "machine code for maximum inference speed on NVIDIA hardware.\n\n"
        "Requirements:\n"
        "  • NVIDIA GPU with CUDA ≥ 11\n"
        "  • TensorRT ≥ 8  (install via pip: tensorrt)\n"
        "  • The exported .engine file is bound to the GPU it was compiled on —\n"
        "    it cannot be transferred to a different GPU model.\n\n"
        "This is an inference-optimisation step, NOT a training format."
    )
    note_box = ctk.CTkTextbox(
        main_frame, font=("Segoe UI", 11), height=145,
        fg_color="#2d2d1e", text_color="#e8e8b0", corner_radius=8,
    )
    note_box.place(relx=0.68, rely=0.38, anchor="center", relwidth=0.54)
    note_box.insert("1.0", _trt_note)
    note_box.configure(state="disabled")

    # Export button
    export_btn = ctk.CTkButton(
        main_frame,
        text="⬇  Export Model",
        command=lambda: export_model(export_fmt_var.get()),
        fg_color="#6a1b9a", hover_color="#4a148c",
        font=("Segoe UI", 15, "bold"), height=50,
        text_color="white", corner_radius=8,
    )
    export_btn.place(relx=0.25, rely=0.58, anchor="center", relwidth=0.32)

    export_status_label = ctk.CTkLabel(
        main_frame, text="", font=("Segoe UI", 12), wraplength=600
    )
    export_status_label.place(relx=0.5, rely=0.72, anchor="center", relwidth=0.85)


# ─────────────────────────────────────────────────────────────────────────────
#  File / folder selection dialogs
# ─────────────────────────────────────────────────────────────────────────────
def select_train_data() -> None:
    global train_data_path, train_data_label
    path = normalize_path(filedialog.askdirectory(title="Select Training Data Folder"))
    if path:
        train_data_path = path
        short = Path(path).name or path
        _safe_label_configure(train_data_label, text=short, text_color="#4caf50")


def select_model_save_folder() -> None:
    global model_save_path, model_save_label
    path = normalize_path(filedialog.askdirectory(title="Select Model Save Folder"))
    if path:
        model_save_path = path
        short = Path(path).name or path
        _safe_label_configure(model_save_label, text=short, text_color="#4caf50")


def select_custom_model() -> None:
    global custom_model_path, custom_model_label
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select Custom Base Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
    )
    if path:
        custom_model_path = path
        _safe_label_configure(
            custom_model_label,
            text=f"Custom: {Path(path).name}",
            text_color="#64b5f6",
        )


def clear_custom_model() -> None:
    global custom_model_path, custom_model_label
    custom_model_path = ""
    _safe_label_configure(
        custom_model_label,
        text="Using built-in pretrained weights",
        text_color="gray",
    )


def select_detection_images_folder() -> None:
    global detection_images_folder_path, detect_folder_label
    path = normalize_path(filedialog.askdirectory(title="Select Images/Videos Folder"))
    if path:
        detection_images_folder_path = path
        short = Path(path).name or path
        _safe_label_configure(detect_folder_label, text=f"Folder: {short}", text_color="#4caf50")


def select_detection_model() -> None:
    global detection_model_path, detect_model_label
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")],
        )
    )
    if path:
        detection_model_path = path
        _safe_label_configure(
            detect_model_label, text=f"Model: {Path(path).name}", text_color="#4caf50"
        )


def select_camera_save_folder() -> None:
    global detection_save_dir
    path = normalize_path(filedialog.askdirectory(title="Select Capture Save Folder"))
    if path:
        detection_save_dir = path
        if camera_detection:
            camera_detection.set_save_directory(path)


def select_export_model() -> None:
    global export_model_path, export_model_label
    path = normalize_path(
        filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
        )
    )
    if path:
        export_model_path = path
        _safe_label_configure(
            export_model_label, text=Path(path).name, text_color="#4caf50"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Training logic
# ─────────────────────────────────────────────────────────────────────────────
def enqueue_output(out, queue: Queue) -> None:
    for line in iter(out.readline, ""):
        queue.put(line)
    out.close()


def start_training(
    project_name_entry,
    input_size_entry,
    epochs_entry,
    batch_size_entry,
    class_names_text,
) -> None:
    global project_name, train_data_path, model_save_path, custom_model_path
    global input_size, epochs, batch_size, class_names

    # Read values from UI widgets
    project_name = project_name_entry.get().strip()
    input_size   = input_size_entry.get().strip()
    epochs_val   = epochs_entry.get().strip()
    batch_val    = batch_size_entry.get().strip()
    raw_classes  = class_names_text.get("1.0", "end-1c")
    class_names  = [n.strip() for n in raw_classes.splitlines() if n.strip()]

    selected_display   = selected_model_var.get() if selected_model_var else ""
    selected_model_size = MODEL_MAP.get(selected_display, "")

    # Validate
    errors: list[str] = []
    if not project_name:
        errors.append("• Project Name is empty.")
    if not train_data_path:
        errors.append("• Training Data Folder not selected.")
    if not model_save_path:
        errors.append("• Model Save Folder not selected.")
    if not selected_model_size and not custom_model_path:
        errors.append("• No YOLO model selected and no custom model loaded.")
    if not input_size or not input_size.isdigit() or int(input_size) < 1:
        errors.append("• Image Size must be a positive integer (e.g. 640).")
    if not epochs_val or not epochs_val.isdigit() or int(epochs_val) < 1:
        errors.append("• Epochs must be a positive integer (e.g. 100).")
    if not batch_val or not batch_val.isdigit() or int(batch_val) < 1:
        errors.append("• Batch Size must be a positive integer (e.g. 16).")
    if not class_names:
        errors.append("• Class Names are empty.")

    if errors:
        messagebox.showerror("Missing / invalid input", "\n".join(errors))
        return

    epochs     = epochs_val
    batch_size = batch_val

    yaml_path = create_yaml(project_name, train_data_path, class_names, model_save_path)
    _run_training_subprocess(yaml_path, selected_model_size)


def _run_training_subprocess(yaml_path: str, selected_model_size: str) -> None:
    """Spawn src/train.py in a subprocess and stream its output to the log box."""
    global progress_bar, output_textbox

    cmd = [
        sys.executable, "src/train.py",
        project_name,
        train_data_path,
        ",".join(class_names),
        model_save_path,
        selected_model_size,
        str(input_size),
        str(epochs),
        yaml_path,
        str(batch_size),
        custom_model_path,   # empty string → train.py treats as "no custom model"
    ]

    def run() -> None:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(
            target=enqueue_output, args=(proc.stdout, output_queue), daemon=True
        ).start()
        proc.wait()
        root.after(0, _training_finished)

    if progress_bar:
        progress_bar.start()
    threading.Thread(target=run, daemon=True).start()


def _training_finished() -> None:
    global progress_bar, output_textbox
    if progress_bar:
        try:
            progress_bar.stop()
        except Exception:
            pass
    try:
        if output_textbox and output_textbox.winfo_exists():
            output_textbox.insert("end", "\n✅ Training process finished.\n")
            output_textbox.yview_moveto(1)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Image / Video detection logic
# ─────────────────────────────────────────────────────────────────────────────
def start_image_detection() -> None:
    global detection_images_folder_path, detection_model_path

    if not detection_images_folder_path:
        messagebox.showerror("Error", "Please select an images/videos folder first.")
        if detection_progress_bar:
            detection_progress_bar.stop()
        return

    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt file) first.")
        if detection_progress_bar:
            detection_progress_bar.stop()
        return

    threading.Thread(
        target=detect_images,
        args=(detection_images_folder_path, detection_model_path, _on_detection_complete),
        daemon=True,
    ).start()


def _on_detection_complete(results_dir: str) -> None:
    """Called from the detection thread; schedules GUI update on the main thread."""
    global image_paths, current_image_index
    image_paths = sorted(
        str(p) for p in Path(results_dir).iterdir()
        if p.is_file() and is_valid_image(str(p))
    )
    current_image_index = 0
    root.after(0, _show_detection_results)


def _show_detection_results() -> None:
    global detection_progress_bar
    if detection_progress_bar:
        try:
            detection_progress_bar.stop()
        except Exception:
            pass
    if image_paths:
        update_image()


# ─────────────────────────────────────────────────────────────────────────────
#  Camera detection logic
# ─────────────────────────────────────────────────────────────────────────────
def start_camera_detection() -> None:
    global camera_detection, image_label, _camera_bar

    if not detection_model_path:
        messagebox.showerror("Error", "Please select a YOLO model (.pt file) first.")
        return

    cam_text = camera_id_entry.get().strip() if camera_id_entry else ""
    if not cam_text:
        messagebox.showerror(
            "Error", "Please enter a Camera ID (e.g. 0 for the default webcam)."
        )
        return

    try:
        camera_id = int(cam_text)
    except ValueError:
        messagebox.showerror("Error", "Camera ID must be an integer (e.g. 0, 1, 2).")
        return

    # Switch button to STOP
    if _camera_bar and hasattr(_camera_bar, "_start_btn"):
        _camera_bar._start_btn.configure(
            text="■  STOP",
            fg_color="#c62828",
            hover_color="#b71c1c",
            command=stop_camera_detection,
        )

    try:
        camera_detection = CameraDetection(detection_model_path)
        camera_detection.start_camera(camera_id)
        if detection_save_dir:
            camera_detection.set_save_directory(detection_save_dir)
        camera_detection.show_camera_stream(image_label)
    except ValueError as exc:
        messagebox.showerror("Camera Error", f"Could not open camera {camera_id}:\n{exc}")
        _reset_camera_button()
    except Exception as exc:
        messagebox.showerror("Error", f"Unexpected error starting camera:\n{exc}")
        _reset_camera_button()


def stop_camera_detection() -> None:
    global camera_detection
    if camera_detection:
        camera_detection.stop()
        camera_detection = None
    _reset_camera_button()


def _reset_camera_button() -> None:
    if _camera_bar and hasattr(_camera_bar, "_start_btn"):
        _camera_bar._start_btn.configure(
            text="▶  START",
            fg_color="#2e7d32",
            hover_color="#1b5e20",
            command=start_camera_detection,
        )


def save_callback() -> None:
    if camera_detection:
        if detection_save_dir:
            camera_detection.set_save_directory(detection_save_dir)
        camera_detection.capture_frame()


# ─────────────────────────────────────────────────────────────────────────────
#  Export logic
# ─────────────────────────────────────────────────────────────────────────────
def export_model(format_display: str) -> None:
    global export_status_label, export_model_path

    if not export_model_path:
        messagebox.showerror("Error", "Please select a trained .pt model to export.")
        return

    fmt = EXPORT_FORMAT_MAP.get(format_display, "onnx")
    _safe_label_configure(
        export_status_label,
        text=f"Exporting to {format_display}…  please wait.",
        text_color="#64b5f6",
    )
    root.update()

    def do_export() -> None:
        try:
            from ultralytics import YOLO
            model = YOLO(export_model_path)
            out = model.export(format=fmt)
            msg = f"✅ Export successful →  {out}"
            root.after(
                0,
                lambda: _safe_label_configure(export_status_label, text=msg, text_color="#4caf50"),
            )
        except Exception as exc:
            err = f"❌ Export failed: {exc}"
            root.after(
                0,
                lambda: _safe_label_configure(export_status_label, text=err, text_color="#ef5350"),
            )

    threading.Thread(target=do_export, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
#  GUI bootstrap
# ─────────────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

screen_w, screen_h = get_screen_size()
root = ctk.CTk()
root.title("YOLO Training & Detection Studio")
root.geometry(f"{screen_w}x{screen_h}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
SIDEBAR_W = 210
sidebar = ctk.CTkFrame(master=root, width=SIDEBAR_W, corner_radius=0, fg_color="#1e1e2e")
sidebar.pack(side="left", fill="y")
sidebar.pack_propagate(False)

ctk.CTkLabel(
    sidebar, text="YOLO Studio", font=("Segoe UI", 17, "bold"), text_color="#cdd6f4"
).pack(pady=(18, 2))
ctk.CTkLabel(
    sidebar, text="Train · Detect · Export", font=("Segoe UI", 10), text_color="#6c7086"
).pack(pady=(0, 6))
ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=(0, 10))

_NAV = [
    ("🏋  Train",   "Train",  "#89b4fa"),
    ("🔍  Detect",  "Detect", "#a6e3a1"),
    ("📷  Camera",  "Camera", "#fab387"),
    ("⬇  Export",  "Export", "#cba6f7"),
]
for _label, _key, _colour in _NAV:
    _btn = ctk.CTkButton(
        sidebar,
        text=_label,
        command=lambda k=_key: on_sidebar_select(k),
        fg_color=_colour,
        text_color="#1e1e2e",
        hover_color="#585b70",
        font=("Segoe UI", 14, "bold"),
        height=46,
        corner_radius=8,
    )
    _btn.pack(fill="x", padx=10, pady=5)

ctk.CTkFrame(sidebar, height=1, fg_color="#45475a").pack(fill="x", padx=10, pady=8)

# Appearance mode toggle
ctk.CTkLabel(
    sidebar, text="Appearance", font=("Segoe UI", 11), text_color="#a6adc8"
).pack(padx=10, anchor="w")
_appearance_var = ctk.StringVar(value="Dark")
ctk.CTkOptionMenu(
    sidebar,
    variable=_appearance_var,
    values=["Light", "Dark", "System"],
    command=ctk.set_appearance_mode,
    font=("Segoe UI", 11),
    height=30,
).pack(fill="x", padx=10, pady=(2, 8))

# Spacer + footer
ctk.CTkLabel(sidebar, text="").pack(fill="both", expand=True)
ctk.CTkLabel(
    sidebar, text="© 2024 SpreadKnowledge", font=("Segoe UI", 9), text_color="#585b70"
).pack(pady=6)

# ── Main frame ────────────────────────────────────────────────────────────────
main_frame = ctk.CTkFrame(master=root, corner_radius=0)
main_frame.pack(fill="both", expand=True)

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root.after(100, update_output_textbox)
    root.mainloop()
