import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout,
                             QHBoxLayout, QFileDialog, QLabel, QWidget, QFrame,
                             QProgressBar, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

MODEL_CANDIDATES = [
  "/Users/kamand/preprocessing/ensemble_output/ensemble_weights.h5"
]
LABELS_PATH = "/Users/kamand/preprocessing/ensemble_output/label_mapping.npy"
MEL_DATA_DIR = "/Users/kamand/preprocessing/midi-features_2_images/mel_spectrograms"
EXPECTED_INPUT_SHAPE = (216, 7, 1)  # here if you're checking with anothe model change the shape, my model and preprocess was ResNet50
TOPK = 5
BAR_COLOR = "#4CAF50" 
def _load_labels(num_outputs: int) -> list:
    if os.path.exists(LABELS_PATH):
        try:
            labels = np.load(LABELS_PATH, allow_pickle=True).tolist()
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            if len(labels) == num_outputs:
                return labels
        except Exception:
            pass
    if os.path.isdir(MEL_DATA_DIR):
        comps = sorted([d for d in os.listdir(MEL_DATA_DIR) if os.path.isdir(os.path.join(MEL_DATA_DIR, d))])
        if len(comps) == num_outputs:
            return comps
    return [f"Class {i}" for i in range(num_outputs)]

#feature extraction 
def extract_mel_feature(y: np.ndarray, sr: int = 22050, n_mels: int = 216, hop_length: int = 512) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    target_time = 7
    if mel_db.shape[1] < target_time:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_time - mel_db.shape[1])), mode='constant')
    elif mel_db.shape[1] > target_time:
        mel_db = mel_db[:, :target_time]
    mel_db = np.expand_dims(mel_db, axis=-1)
    # Debug: print shape of mel_db before returning
    print(f"[DEBUG] Mel feature shape after padding/trimming: {mel_db.shape}")
    return mel_db.astype(np.float32)
#model architecture + user interface
class ComposerAnalysisModel:
    def __init__(self):
        model_path = None
        for cand in MODEL_CANDIDATES:
            if os.path.exists(cand):
                model_path = cand
                break
        if model_path is None:
            raise FileNotFoundError("No trained mel-only model found. Checked: " + "".join(MODEL_CANDIDATES))

        self.model = models.load_model(model_path, compile=False)
        num_outputs = int(self.model.output_shape[-1])
        self.composers = _load_labels(num_outputs)
        if len(self.composers) != num_outputs:
            self.composers = [f"Class {i}" for i in range(num_outputs)]

    def _segment_audio(self, y: np.ndarray, sr: int, win_sec=8.0, hop_sec=4.0):
        win = int(win_sec * sr)
        hop = int(hop_sec * sr)
        if len(y) <= win:
            yield y
            return
        for start in range(0, len(y) - win + 1, hop):
            yield y[start:start + win]

    def _batch_from_audio(self, y: np.ndarray, sr: int):
        mel_imgs = []
        any_seg = False
        for seg in self._segment_audio(y, sr):
            any_seg = True
            mel_feature = extract_mel_feature(seg, sr)
            mel_imgs.append(mel_feature)
        if not any_seg:
            mel_imgs.append(extract_mel_feature(y, sr))
        mel_stack = np.stack(mel_imgs)
        # Debug: print stacked shape
        print(f"[DEBUG] Mel batch shape: {mel_stack.shape}")
        return mel_stack

    def process_audio(self, audio_path: str):
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        print(f"[DEBUG] Loaded audio: {audio_path}, length={len(y)}, sr={sr}")
        mel_b = self._batch_from_audio(y, sr)
        probs = self.model.predict(mel_b, verbose=0)
        print(f"[DEBUG] Model prediction shape: {probs.shape}")
        probs_mean = probs.mean(axis=0)
        order = np.argsort(probs_mean)[::-1]
        top_idx = order[:TOPK]
        top_composers = [self.composers[i] for i in top_idx]
        top_scores = probs_mean[top_idx].tolist()
        print(f"[DEBUG] Top composers: {top_composers}, scores: {top_scores}")
        return top_composers, top_scores


class ProcessingThread(QThread):
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(int)

    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path

    def run(self):
        try:
            self.progress.emit(30)
            top_composers, top_scores = self.model.process_audio(self.audio_path)
            self.progress.emit(100)
            self.finished.emit(top_composers, top_scores)
        except Exception as e:
            self.progress.emit(100)
            self.finished.emit([f"Error: {e}"], [1.0])

#PyQt look of the python app
class ComposerSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.model = ComposerAnalysisModel()
        except Exception as e:
            self.model = None
            QMessageBox.critical(self, "Model Load Error", str(e))
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Composer Similarity Analysis")
        self.setMinimumSize(800, 600)

        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        title_label = QLabel("Composer Similarity Analysis")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        desc_label = QLabel(
            "Upload an audio file to analyze which composers' styles it most resembles. "
            "The system uses a deep learning model trained on classical piano compositions."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)

        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 5px;")

        browse_button = QPushButton("Browse")
        browse_button.setFixedWidth(100)
        browse_button.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(browse_button)
        main_layout.addLayout(file_layout)

        self.analyze_button = QPushButton("Analyze Composition")
        self.analyze_button.setFixedHeight(40)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_audio)
        main_layout.addWidget(self.analyze_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")

        results_layout = QVBoxLayout(results_frame)

        results_title = QLabel("Analysis Results")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_title)

        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)

        self.update_plot([], [])

        main_layout.addWidget(results_frame, 1)
        self.setCentralWidget(main_widget)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.webm)"
        )
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.file_path = file_path
            self.analyze_button.setEnabled(True and (self.model is not None))

    def analyze_audio(self):
        if self.model is None:
            QMessageBox.warning(self, "Model not loaded", "Please fix model path and restart.")
            return
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analyze_button.setEnabled(False)

        self.processing_thread = ProcessingThread(self.model, self.file_path)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.display_results)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def display_results(self, composers, scores):
        if composers and isinstance(composers[0], str) and composers[0].startswith("Error:"):
            QMessageBox.critical(self, "Processing Error", composers[0])
            self.progress_bar.setVisible(False)
            self.analyze_button.setEnabled(True)
            return

        self.update_plot(composers, scores)
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)

    def update_plot(self, composers, scores):
        self.figure.clear()
        if composers and scores:
            ax = self.figure.add_subplot(111)
            y_pos = np.arange(len(composers))
            ax.barh(y_pos, scores, align='center', color=BAR_COLOR, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(composers)
            ax.invert_yaxis()
            ax.set_xlabel('Similarity Score')
            ax.set_title('Top 5 Similar Composers')
            for i, v in enumerate(scores):
                ax.text(v + 0.01, i, f"{v:.1%}", va='center')
            ax.set_xlim(0, max(scores) * 1.2 if len(scores) else 1.0)
            self.figure.tight_layout()
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Upload and analyze an audio file to see results",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = ComposerSimilarityApp()
    window.show()
    sys.exit(app.exec_())
