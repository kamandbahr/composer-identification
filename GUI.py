import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, 
                            QHBoxLayout, QFileDialog, QLabel, QWidget, QFrame,
                            QProgressBar, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

class ComposerAnalysisModel:
    """
        Here the actual model should exist.
 
    """
    def __init__(self):
        # In a real implementation, load model weights here
        self.composers = ["Bach", "Mozart", "Beethoven", "Chopin", "Debussy", 
                         "Rachmaninoff", "Liszt", "Schubert", "Tchaikovsky", "Brahms"]
        
    def process_audio(self, audio_path):
 
        
        print(f"Processing audio: {audio_path}")
        
        import time
        time.sleep(2)
        
        
        # the similarity is fake but the sum will be 100 at the end
        similarities = np.random.random(len(self.composers))
        similarities = similarities / np.sum(similarities)  # Normalize to sum to 1
        
        # sort composers by similarity score
        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        top_composers = [self.composers[i] for i in sorted_indices[:5]]
        top_scores = [similarities[i] for i in sorted_indices[:5]]
        
        return top_composers, top_scores

class ProcessingThread(QThread):
    """Thread for running audio processing without freezing the UI"""
    finished = pyqtSignal(list, list)
    progress = pyqtSignal(int)
    
    def __init__(self, model, audio_path):
        super().__init__()
        self.model = model
        self.audio_path = audio_path
        
    def run(self):
        self.progress.emit(30)
        top_composers, top_scores = self.model.process_audio(self.audio_path)
        self.progress.emit(100)
        self.finished.emit(top_composers, top_scores)

class ComposerSimilarityApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.model = ComposerAnalysisModel()
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Composer Similarity Analysis")
        self.setMinimumSize(800, 600)
        
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # title
        title_label = QLabel("Composer Similarity Analysis")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # description
        desc_label = QLabel(
            "Upload an audio file to analyze which composers' styles it most resembles. "
            "The system uses a deep learning model trained on classical piano compositions."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(desc_label)
        
        # here the line should be horizontal 
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(line)
        
        # create the file selection area
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
        
        browse_button = QPushButton("Browse")
        browse_button.setFixedWidth(100)
        browse_button.clicked.connect(self.browse_file)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(browse_button)
        
        main_layout.addLayout(file_layout)
        
        # analyze button
        self.analyze_button = QPushButton("Analyze Composition")
        self.analyze_button.setFixedHeight(40)
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_audio)
        main_layout.addWidget(self.analyze_button)
        
        # progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # results area
        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet("background-color: #f9f9f9; border-radius: 5px;")
        
        results_layout = QVBoxLayout(results_frame)
        
        results_title = QLabel("Analysis Results")
        results_title.setFont(QFont("Arial", 14, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_title)
        
        # figure for the plot
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        results_layout.addWidget(self.canvas)
   
        self.update_plot([], [])
        
        main_layout.addWidget(results_frame, 1)  # 1 means this widget takes available space
        self.setCentralWidget(main_widget)
        
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "", 
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a *.webm)"
        )
        
        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.file_path = file_path
            self.analyze_button.setEnabled(True)
    
    def analyze_audio(self):
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analyze_button.setEnabled(False)
        
        # start processing part
        self.processing_thread = ProcessingThread(self.model, self.file_path)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.display_results)
        self.processing_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def display_results(self, composers, scores):
        # adding results
        self.update_plot(composers, scores)
        
        # after the process is done again the button should be available
        self.progress_bar.setVisible(False)
        self.analyze_button.setEnabled(True)
    
    def update_plot(self, composers, scores):
        # clear the figure
        self.figure.clear()
        
        # bar chart
        if composers and scores:
            ax = self.figure.add_subplot(111)
            
            # horizontal bar chart
            y_pos = np.arange(len(composers))
            ax.barh(y_pos, scores, align='center', color='#4CAF50', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(composers)
            ax.invert_yaxis()  # Labels read top-to-bottom pay attention to here!!
            ax.set_xlabel('Similarity Score')
            ax.set_title('Top 5 Similar Composers')
            
            for i, v in enumerate(scores):
                ax.text(v + 0.01, i, f"{v:.1%}", va='center')
                
            ax.set_xlim(0, max(scores) * 1.2)
            
            self.figure.tight_layout()
        else:
            # if no data show the first window!
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Upload and analyze an audio file to see results", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.axis('off')
        
        #refresh the canvas here
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion') 
    window = ComposerSimilarityApp()
    window.show()
    sys.exit(app.exec_())