import os
import pathlib
import sys
from collections import deque
from pathlib import Path
from typing import List, Optional

import PySide6

# Force PySide6 plugin path first to avoid cv2/conda Qt conflicts
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
_pyside_root = pathlib.Path(PySide6.__file__).resolve().parent
_qt_plugins = _pyside_root / "Qt" / "plugins"
_qt_platforms = _qt_plugins / "platforms"
os.environ["QT_PLUGIN_PATH"] = str(_qt_plugins)
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_platforms)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import mediapipe as mp
import numpy as np
import torch
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from model.cnn_lstm import CNN_LSTM


ACTIONS = [
    "Select Drone", "Select Group", "Select Mode", "ARM", "DISARM", "TAKEOFF",
    "LAND", "RTL", "Change Altitude", "Change Speed", "Move Up", "Move Down",
    "Rotate CW", "Move Forward", "Move Backward", "Move Right", "Move Left",
    "Rotate CCW", "Cancel", "Check", "One", "Two", "Three", "Four", "Five",
    "Six", "Seven", "Eight", "Nine", "Ten",
]


# 카메라 인덱스 중 사용 가능한 카메라 리스트
def find_available_cameras(max_index: int = 10) -> List[int]:
    available = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        ok, _ = cap.read()
    
        if cap.isOpened() and ok:
            available.append(index)
        cap.release()
    
    return available


# 제스처 인식 로직
class GestureEngine:
    def __init__(self, model_path: str = "./hmi_FGCS.pt"):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN_LSTM(
            input_size=99,
            output_size=128,
            hidden_size=64,
            num_classes=30,
        ).to(self.device)

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.seq_length = 30
        self.seq = []
        self.action_seq = []
        self.thres_fr = 30
        self.current_action = "Waiting"

    
    # 한 프레임의 손 랜드마크, 현재 제스처 반환(30 프레임 이상 쌓이면 모델 추론 진행)
    def process(self, frame_bgr: np.ndarray) -> dict:
        img = cv2.flip(frame_bgr, 1)
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        detected_hands = 0
        if result.multi_hand_landmarks is not None:
            detected_hands = len(result.multi_hand_landmarks)
            for res in result.multi_hand_landmarks:
                joint = np.zeros((21, 4), dtype=np.float32)
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]

                v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
                v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
                v = v2 - v1
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                norm = np.where(norm == 0, 1e-6, norm)
                v = v / norm

                angle = np.arccos(np.clip(np.einsum(
                    "nt,nt->n",
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18], :],
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19], :],
                ), -1.0, 1.0))
                angle = np.degrees(angle)

                gesture_joint = np.concatenate([joint.flatten(), angle]).astype(np.float32)
                self.seq.append(gesture_joint)

                self.mp_drawing.draw_landmarks(
                    img,
                    res,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                if len(self.seq) < self.seq_length:
                    continue

                input_data = np.expand_dims(np.array(self.seq[-self.seq_length:], dtype=np.float32), axis=0)
                input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    y_pred = self.model(input_data)
                    values, indices = torch.max(y_pred, dim=1)

                model_confidence = float(values.item())
                if model_confidence < 0.9:
                    continue

                pred_idx = int(indices.item())
                action = ACTIONS[pred_idx]
                self.action_seq.append(action)

                if len(self.action_seq) < self.thres_fr:
                    continue

                if all(a == self.action_seq[-1] for a in self.action_seq[-self.thres_fr:]):
                    self.current_action = self.action_seq[-1]
                    self.action_seq = []

        return {
            "frame": img,
            "current_action": self.current_action,
        }


# 카메라 읽기, 추론을 백그라운드에서 진행
class CameraWorker(QThread):
    frame_ready = Signal(object)
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, camera_index: int, model_path: str = "./hmi_FGCS.pt"):
        super().__init__()
        self.camera_index = camera_index
        self.model_path = model_path
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.engine: Optional[GestureEngine] = None

    def run(self):
        self.running = True
        try:
            self.engine = GestureEngine(model_path=self.model_path)
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                self.error_occurred.emit(f"Camera {self.camera_index} could not be opened")
                return

            self.status_changed.emit("Live")
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    self.error_occurred.emit("Failed to read frame from camera")
                    break
                self.frame_ready.emit(self.engine.process(frame))
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self.status_changed.emit("Stopped")
            if self.cap is not None:
                self.cap.release()

    def stop(self):
        self.running = False
        self.wait(1000)


# 정보 출력 UI
class InfoPanel(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("InfoPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        self.title_label = QLabel("Current Gesture")
        self.title_label.setObjectName("SectionLabel")
        self.action_label = QLabel("Waiting")
        self.action_label.setObjectName("ActionLabel")

        self.status_title = QLabel("Status")
        self.status_title.setObjectName("SectionLabel")
        self.status_label = QLabel("Idle")
        self.status_label.setObjectName("ValueLabel")

        self.camera_title = QLabel("Camera")
        self.camera_title.setObjectName("SectionLabel")
        self.camera_label = QLabel("-")
        self.camera_label.setObjectName("ValueLabel")

        self.message_title = QLabel("Message")
        self.message_title.setObjectName("SectionLabel")
        self.message_label = QLabel("Ready")
        self.message_label.setObjectName("MessageLabel")
        self.message_label.setWordWrap(True)

        for widget in [
            self.title_label, self.action_label,
            self.status_title, self.status_label,
            self.camera_title, self.camera_label,
            self.message_title, self.message_label,
        ]:
            layout.addWidget(widget)
        layout.addStretch(1)


# 전체 UI                
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture UI")
        self.resize(1280, 820)
        self.worker: Optional[CameraWorker] = None

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        left_layout = QVBoxLayout()
        left_layout.setSpacing(12)
        right_layout = QVBoxLayout()
        right_layout.setSpacing(12)

        main_layout.addLayout(left_layout, 5)
        main_layout.addLayout(right_layout, 2)

        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 14, 14, 14)
        top_layout.setSpacing(10)

        self.camera_select = QComboBox()
        self.available_cameras = find_available_cameras()
        for index in self.available_cameras:
            self.camera_select.addItem(f"Camera {index}", index)
        if not self.available_cameras:
            self.camera_select.addItem("No camera found", None)
            self.camera_select.setEnabled(False)

        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(bool(self.available_cameras))

        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)

        top_layout.addWidget(self.camera_select)
        top_layout.addStretch(1)
        top_layout.addWidget(self.start_button)
        top_layout.addWidget(self.stop_button)
        left_layout.addWidget(top_bar)

        self.video_label = QLabel("Camera feed")
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(840, 640)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout.addWidget(self.video_label, 1)

        self.info_panel = InfoPanel()
        right_layout.addWidget(self.info_panel)
        right_layout.addStretch(1)

        self.apply_styles()

    def apply_styles(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background: #111111;
            }
            QFrame#TopBar, QFrame#InfoPanel {
                background: #181818;
                border: 1px solid #262626;
                border-radius: 12px;
            }
            QLabel {
                color: #d4d4d4;
            }
            QLabel#VideoLabel {
                background: #0d0d0d;
                border: 1px solid #262626;
                border-radius: 12px;
                color: #777777;
                font-size: 18px;
            }
            QLabel#SectionLabel {
                color: #888888;
                font-size: 12px;
            }
            QLabel#ActionLabel {
                color: #f2f2f2;
                font-size: 28px;
                font-weight: 700;
                padding-bottom: 8px;
            }
            QLabel#ValueLabel {
                color: #f2f2f2;
                font-size: 18px;
                font-weight: 600;
                padding-bottom: 8px;
            }
            QLabel#MessageLabel {
                color: #cfcfcf;
                font-size: 14px;
                line-height: 1.4;
            }
            QComboBox, QPushButton {
                background: #1b1b1b;
                color: #e8e8e8;
                border: 1px solid #2e2e2e;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 14px;
            }
            QPushButton:disabled {
                color: #737373;
                background: #141414;
                border: 1px solid #222222;
            }
            """
        )

    def start_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return

        cam_index = self.camera_select.currentData()
        if cam_index is None:
            self.info_panel.message_label.setText("No available camera detected")
            return

        self.worker = CameraWorker(camera_index=cam_index, model_path=str(Path("./hmi_FGCS.pt")))
        self.info_panel.camera_label.setText(f"Camera {cam_index}")
        self.worker.frame_ready.connect(self.update_ui)
        self.worker.status_changed.connect(self.update_status)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.info_panel.message_label.setText("Initializing")

    def stop_camera(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.start_button.setEnabled(bool(self.available_cameras))
        self.stop_button.setEnabled(False)
        self.info_panel.status_label.setText("Stopped")
        self.info_panel.camera_label.setText("-")
        self.info_panel.message_label.setText("Stream stopped")

    def update_status(self, status: str):
        self.info_panel.status_label.setText(status)

    def show_error(self, message: str):
        self.info_panel.status_label.setText("Error")
        self.info_panel.message_label.setText(message)
        self.start_button.setEnabled(bool(self.available_cameras))
        self.stop_button.setEnabled(False)

    def update_ui(self, result: dict):
        frame = result["frame"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        self.info_panel.action_label.setText(result["current_action"])
        self.info_panel.message_label.setText("Running")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.video_label.pixmap() is not None:
            self.video_label.setPixmap(
                self.video_label.pixmap().scaled(
                    self.video_label.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Sans Serif", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
