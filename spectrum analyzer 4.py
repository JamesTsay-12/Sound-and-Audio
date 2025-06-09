import sys
import csv
import time
import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import aubio
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import threading
import scipy.io.wavfile as wav
from PyQt5.QtWidgets import QFileDialog

# Constants
fs = 44100
buffer_size = 1024
hop_size = buffer_size // 2
waterfall_length = 200
freqs = np.fft.rfftfreq(buffer_size, 1 / fs)

pitch_o = aubio.pitch("yin", buffer_size, hop_size, fs)
pitch_o.set_unit("Hz")
pitch_o.set_silence(-40)


class AudioFileProcessor(QtCore.QThread):
    chunk_processed = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, audio_data, hop_size):
        super().__init__()
        self.audio_data = audio_data
        self.hop_size = hop_size
        self._running = True

    def run(self):
        length = len(self.audio_data)
        i = 0
        while i < length and self._running:
            chunk = self.audio_data[i:i + self.hop_size]
            if len(chunk) < self.hop_size:
                chunk = np.pad(chunk, (0, self.hop_size - len(chunk)), 'constant')
            self.chunk_processed.emit(chunk)
            i += self.hop_size
            self.msleep(10)

    def stop(self):
        self._running = False
        self.wait()


class AudioAnalyzer(QtWidgets.QMainWindow):
    def __init__(self, device_index=None, audio_file=None):
        super().__init__()
        self.device_index = device_index
        self.audio_file = audio_file
        self.recording = False
        self.audio_data = []
        self.audio_file_thread = None
        self.initUI()
        self.waterfall_data = np.zeros((waterfall_length, buffer_size // 2 + 1))
        self.start_time = time.time()
        self.csv_file = None
        self.csv_writer = None
        self.stream = None

        if audio_file:
            self.load_audio_file(audio_file)
        else:
            self.setup_audio_stream()

        # QTimer to update spectrogram every 2 seconds
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_spectrogram)
        self.timer.start(2000)

    def initUI(self):
        self.setWindowTitle("Vocal Spectrum Analyzer")
        self.resize(1000, 700)
        cw = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()
        cw.setLayout(layout)
        self.setCentralWidget(cw)

        self.pitch_label = QtWidgets.QLabel("Pitch: --- Hz (---)")
        layout.addWidget(self.pitch_label)

        self.waveform_plot = pg.PlotWidget(title="Waveform")
        self.waveform_curve = self.waveform_plot.plot(pen='y')
        layout.addWidget(self.waveform_plot)

        self.spectrum_plot = pg.PlotWidget(title="Spectrum (Log X)")
        self.spectrum_curve = self.spectrum_plot.plot(pen='c')
        self.spectrum_plot.setLogMode(x=True, y=False)
        self.spectrum_plot.setYRange(0, 100)
        layout.addWidget(self.spectrum_plot)

        self.waterfall_img = pg.ImageView()
        layout.addWidget(self.waterfall_img)
        self.waterfall_img.ui.histogram.hide()
        self.waterfall_img.ui.roiBtn.hide()
        self.waterfall_img.ui.menuBtn.hide()

        # Matplotlib spectrogram canvas embedded in UI
        self.fig = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Live Spectrogram")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Frequency [Hz]")

        button_layout = QtWidgets.QHBoxLayout()
        self.toggle_button = QtWidgets.QPushButton("Start Recording")
        self.toggle_button.clicked.connect(self.toggle_recording)
        button_layout.addWidget(self.toggle_button)

        self.load_button = QtWidgets.QPushButton("Load .wav File")
        self.load_button.clicked.connect(self.load_file)
        button_layout.addWidget(self.load_button)

        self.exit_button = QtWidgets.QPushButton("Stop & Exit")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)

        layout.addLayout(button_layout)

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.recording = True
        self.toggle_button.setText("Pause Recording")
        self.csv_file = open("spectrum_export.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp (s)'] + [f'{int(f)} Hz' for f in freqs])
        if self.stream:
            self.stream.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.toggle_button.setText("Start Recording")
            if self.stream:
                try:
                    self.stream.stop()
                except Exception as e:
                    print(f"Error stopping stream: {e}")
            if self.csv_file:
                try:
                    self.csv_file.close()
                except Exception as e:
                    print(f"Error closing CSV file: {e}")
                self.csv_file = None
                self.csv_writer = None

    def closeEvent(self, event):
        self.stop_recording()

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping/closing stream on exit: {e}")
            self.stream = None

        if self.audio_file_thread and self.audio_file_thread.isRunning():
            self.audio_file_thread.stop()
            self.audio_file_thread = None

        event.accept()

    def setup_audio_stream(self):
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=fs,
            blocksize=hop_size,
            device=self.device_index
        )

    def load_audio_file(self, audio_file):
        rate, data = wav.read(audio_file)
        if data.ndim > 1:
            data = data[:, 0]
        self.audio_data = data.tolist()
        self.stream = None  # disable live stream

        if self.audio_file_thread and self.audio_file_thread.isRunning():
            self.audio_file_thread.stop()

        self.audio_file_thread = AudioFileProcessor(np.array(self.audio_data), hop_size)
        self.audio_file_thread.chunk_processed.connect(self.process_audio)
        self.audio_file_thread.start()

    def audio_callback(self, indata, frames, time_info, status):
        samples = indata[:, 0]
        self.process_audio(samples)

    def process_audio(self, samples):
        self.audio_data.extend(samples)
        self.waveform_curve.setData(np.array(samples))

        spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
        spectrum_db = 20 * np.log10(np.maximum(spectrum, 1e-6))
        self.spectrum_curve.setData(freqs, spectrum_db)

        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        self.waterfall_data[-1, :] = spectrum_db
        self.waterfall_img.setImage(self.waterfall_data.T, autoLevels=False)
        self.waterfall_img.setPredefinedGradient('thermal')

        pitch = pitch_o(np.array(samples, dtype=np.float32))[0]
        if 50 < pitch < 2000:
            note = aubio.freq2note(pitch)
            self.pitch_label.setText(f"Pitch: {pitch:.1f} Hz ({note})")
        else:
            self.pitch_label.setText("Pitch: --- Hz (---)")

        if self.recording and self.csv_writer:
            timestamp = time.time() - self.start_time
            row = [f"{timestamp:.3f}"] + [f"{v:.2f}" for v in spectrum_db]
            self.csv_writer.writerow(row)

    def update_spectrogram(self):
        if len(self.audio_data) < fs:
            return

        data = np.array(self.audio_data[-fs * 5:])  # last 5 seconds
        self.ax.clear()
        self.ax.specgram(data, NFFT=1024, Fs=fs, noverlap=512, cmap="plasma")
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Frequency [Hz]")
        self.ax.set_title("Live Spectrogram")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def load_file(self):
        options = QFileDialog.Options()
        file, _ = QFileDialog.getOpenFileName(self, "Open Audio File", "", "WAV Files (*.wav);;All Files (*)", options=options)
        if file:
            self.load_audio_file(file)
            self.update_ui_for_file()

    def update_ui_for_file(self):
        self.pitch_label.setText("Pitch: --- Hz (---)")
        self.toggle_button.setEnabled(False)
        self.load_button.setEnabled(False)


class DeviceSelector(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Audio Input Device")
        self.layout = QtWidgets.QVBoxLayout(self)

        self.device_list = sd.query_devices()
        self.input_devices = [(i, d['name']) for i, d in enumerate(self.device_list) if d['max_input_channels'] > 0]

        self.combo = QtWidgets.QComboBox()
        for index, name in self.input_devices:
            self.combo.addItem(f"{name}", index)
        self.layout.addWidget(QtWidgets.QLabel("Choose Input Device:"))
        self.layout.addWidget(self.combo)

        self.ok_button = QtWidgets.QPushButton("Start Analyzer")
        self.ok_button.clicked.connect(self.accept)
        self.layout.addWidget(self.ok_button)

    def get_selected_device(self):
        return self.combo.currentData()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    selector = DeviceSelector()

    if selector.exec_() == QtWidgets.QDialog.Accepted:
        selected_device = selector.get_selected_device()
        analyzer = AudioAnalyzer(device_index=selected_device)
        analyzer.show()
        sys.exit(app.exec_())
