import queue
import threading

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from rtlsdr import RtlSdr
import sounddevice as sd
from scipy.signal import butter, sosfilt

# SDR setup
sdr = RtlSdr()
sdr.sample_rate = 2.4e6 # controls span of data 
sdr.center_freq = 100.0 * 1e6 # FM range 
sdr.gain = "auto"

# Options
FFT_SIZE = 8192 # the frequency resolution is sample_rate / FFT_SIZE
WATERFALL_ROWS = 120 # how much history to keep on screen
POWER_MIN = -120.0 # should be around the noise floor (min estimate of noise power in dB)
POWER_MAX = -40.0

UPDATE_INTERVAL = 100
AUDIO_RATE = 48e3 # 48kHz audio
READ_SIZE = 131072 # multiple of fft size

DECIMATION = int(sdr.sample_rate // AUDIO_RATE)

audio_filter = butter(5, 15_000, btype="lowpass", fs=sdr.sample_rate, output="sos")
audio_filter_state = np.zeros((audio_filter.shape[0], 2))
audio_resample_buffer = np.empty(0, dtype=np.float64)
audio_queue = queue.Queue(maxsize=8)
audio_pending = np.empty(0, dtype=np.float32)
display_queue = queue.Queue(maxsize=2)


def audio_callback(outdata, frames, _time_info, _status):
    global audio_pending

    while audio_pending.size < frames:
        try:
            chunk = audio_queue.get_nowait()
        except queue.Empty:
            break
        audio_pending = np.concatenate((audio_pending, chunk))

    outdata.fill(0)
    count = min(frames, audio_pending.size)
    if count:
        outdata[:count, 0] = audio_pending[:count]
        audio_pending = audio_pending[count:]

audio_stream = sd.OutputStream(
    samplerate=AUDIO_RATE,
    channels=1,
    dtype="float32",
    callback=audio_callback,
)
audio_stream.start()

window = np.hanning(FFT_SIZE) # filter out signal leakage across bins
# assign frequencies for each bin
freqs_raw = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / sdr.sample_rate) + sdr.center_freq)
freqs = freqs_raw / 1e6 # MHz 

# setup
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(win)

# plotting the spectrum
waterfall = np.full((WATERFALL_ROWS, FFT_SIZE), -160.0)
spectrum_plot = pg.PlotWidget(labels={"left": "PSD (dB/Hz)", "bottom": "Frequency (MHz)"})
spectrum_plot.setMouseEnabled(x=False, y=False) # get rid of default dragging functionality
curve = spectrum_plot.plot(freqs, waterfall[-1], pen=pg.mkPen("y", width=1))
spectrum_plot.setYRange(POWER_MIN, POWER_MAX)
spectrum_plot.getPlotItem().enableAutoRange(False, False)

# adding spectrum plot to layout
top_row = QtWidgets.QHBoxLayout()
top_row.addWidget(spectrum_plot)
top_right_spacer = QtWidgets.QWidget() # need to add a spacer to the right to make sure it aligns with the waterfall
top_right_spacer.setFixedWidth(120)
top_row.addWidget(top_right_spacer)
layout.addLayout(top_row)

# just testing the frequency range selector thign
region = pg.LinearRegionItem([freqs[2048], freqs[6144]], brush=(100, 100, 150, 50))
region.setBounds((freqs[0], freqs[-1]))
spectrum_plot.addItem(region)

selection_offset = 0.0
selection_phase = 0.0
selection_filter = None
selection_filter_state = None
DEMOD_MODE = "FM"


def configure_selection():
    global selection_offset, selection_phase, selection_filter
    global selection_filter_state, audio_filter_state
    global audio_resample_buffer

    low_mhz, high_mhz = region.getRegion()
    bandwidth = max((high_mhz - low_mhz) * 1e6, 20_000.0)
    cutoff = min(bandwidth / 2.0, sdr.sample_rate * 0.45)
    selection_filter = butter(
        5, cutoff, btype="lowpass", fs=sdr.sample_rate, output="sos"
    )
    selection_filter_state = np.zeros(
        (selection_filter.shape[0], 2), dtype=complex
    )
    selection_offset = ((low_mhz + high_mhz) / 2.0) * 1e6 - sdr.center_freq
    selection_phase = 0.0
    audio_filter_state.fill(0)
    audio_resample_buffer = np.empty(0, dtype=np.float64)


configure_selection()
region.sigRegionChanged.connect(configure_selection)

# plotting the waterfall
waterfall_plot = pg.PlotWidget(labels={"left": "Time", "bottom": "Frequency (MHz)"})
waterfall_plot.setMouseEnabled(x=False, y=False)
waterfall_plot.invertY(True)
image = pg.ImageItem(waterfall, axisOrder="row-major")  
image.setRect(
    QtCore.QRectF(
        freqs[0],
        0,
        freqs[-1] - freqs[0],
        WATERFALL_ROWS,
    )
)
waterfall_plot.addItem(image)
waterfall_plot.setXLink(spectrum_plot)
waterfall_plot.setYRange(0, WATERFALL_ROWS)

axis_width = 80
spectrum_plot.getPlotItem().getAxis("left").setWidth(axis_width)
waterfall_plot.getPlotItem().getAxis("left").setWidth(axis_width)

# adding waterfall to layout
waterfall_row = QtWidgets.QHBoxLayout()
waterfall_row.addWidget(waterfall_plot)

# making the colour bar
colorbar = pg.HistogramLUTWidget()
colorbar.setImageItem(image)
colorbar.item.gradient.loadPreset("plasma") 
image.setLevels((POWER_MIN, POWER_MAX)) # needs to come after colorbar is created for some reason
colorbar.orientation = "vertical"
colorbar.setFixedWidth(120)
waterfall_row.addWidget(colorbar)
layout.addLayout(waterfall_row)

# IQ scatter and phase plots
iq_plot = pg.PlotWidget(labels={"left": "Q", "bottom": "I"})
iq_plot.setMouseEnabled(x=False, y=False)
iq_plot.setAspectLocked(True, ratio=1)
iq_plot.getPlotItem().enableAutoRange(False, False)
iq_plot.setXRange(-1.0, 1.0)
iq_plot.setYRange(-1.0, 1.0)
iq_scatter = pg.ScatterPlotItem(size=2, pen=None, brush=pg.mkBrush(0, 255, 0, 120))
iq_plot.addItem(iq_scatter)

phase_plot = pg.PlotWidget(labels={"left": "Phase (rad)", "bottom": "Time (s)"})
phase_plot.setMouseEnabled(x=False, y=False)
phase_curve = phase_plot.plot([], [], pen=pg.mkPen("c", width=1))

iq_phase_row = QtWidgets.QHBoxLayout()
iq_phase_row.addWidget(iq_plot)
iq_phase_row.addWidget(phase_plot)
layout.addLayout(iq_phase_row)

# window setup
win.setWindowTitle("RTL-SDR Spectrum")
win.showMaximized()

def power_spectral_density(samples):
    spectrum = np.fft.fft(samples * window)

    # psd estimate (power per Hz)
    psd = (np.abs(spectrum) ** 2) / (sdr.sample_rate * np.sum(window ** 2)) # np.sum(window**2) corrects for the window energy
    psd_db = 10 * np.log10(psd + 1e-30) # add 1e-30 to avoid log(0)
    psd_shifted = np.fft.fftshift(psd_db)
    return psd_shifted


def fm_demodulate_audio(samples):
    global audio_filter_state, audio_resample_buffer
    global selection_phase, selection_filter_state

    sample_indexes = np.arange(samples.size)
    mixer = np.exp(
        1j * (selection_phase - 2.0 * np.pi * selection_offset * sample_indexes / sdr.sample_rate)
    )
    selection_phase = (
        selection_phase
        - 2.0 * np.pi * selection_offset * samples.size / sdr.sample_rate
    ) % (2.0 * np.pi)
    selected_samples, selection_filter_state = sosfilt(
        selection_filter,
        samples * mixer,
        zi=selection_filter_state,
    )

    phase_differences = np.angle(
        selected_samples[1:] * np.conj(selected_samples[:-1])
    )
    filtered, audio_filter_state = sosfilt(
        audio_filter, phase_differences, zi=audio_filter_state
    )
    audio_resample_buffer = np.concatenate((audio_resample_buffer, filtered))

    usable = audio_resample_buffer.size - audio_resample_buffer.size % DECIMATION
    audio = audio_resample_buffer[:usable:DECIMATION]
    audio_resample_buffer = audio_resample_buffer[usable:]
    return np.clip(audio * 4.0, -1.0, 1.0).astype(np.float32)


def am_demodulate_audio(samples):
    global audio_filter_state, audio_resample_buffer
    global selection_phase, selection_filter_state

    sample_indexes = np.arange(samples.size)
    mixer = np.exp(
        1j * (selection_phase - 2.0 * np.pi * selection_offset * sample_indexes / sdr.sample_rate)
    )
    selection_phase = (
        selection_phase
        - 2.0 * np.pi * selection_offset * samples.size / sdr.sample_rate
    ) % (2.0 * np.pi)
    selected_samples, selection_filter_state = sosfilt(
        selection_filter,
        samples * mixer,
        zi=selection_filter_state,
    )

    envelope = np.abs(selected_samples)
    filtered, audio_filter_state = sosfilt(
        audio_filter, envelope, zi=audio_filter_state
    )
    audio_resample_buffer = np.concatenate((audio_resample_buffer, filtered))

    usable = audio_resample_buffer.size - audio_resample_buffer.size % DECIMATION
    audio = audio_resample_buffer[:usable:DECIMATION]
    audio_resample_buffer = audio_resample_buffer[usable:]
    return np.clip(audio * 4.0, -1.0, 1.0).astype(np.float32)


def demodulate_audio(samples, mode=DEMOD_MODE):
    if mode.upper() == "AM":
        return am_demodulate_audio(samples)
    return fm_demodulate_audio(samples)


def capture_callback(samples, _context):
    audio = demodulate_audio(samples, mode=DEMOD_MODE)
    if audio.size:
        try:
            audio_queue.put_nowait(audio)
        except queue.Full:
            pass

    try:
        display_queue.put_nowait(samples[:FFT_SIZE].copy())
    except queue.Full:
        try:
            display_queue.get_nowait()
            display_queue.put_nowait(samples[:FFT_SIZE].copy())
        except queue.Empty:
            pass


def capture_loop():
    sdr.read_samples_async(capture_callback, num_samples=READ_SIZE)


def update():
    global waterfall

    try:
        fft_samples = display_queue.get_nowait()
    except queue.Empty:
        return

    # calculate data for IQ and phase plots
    i = np.real(fft_samples)
    q = np.imag(fft_samples)
    phase = np.angle(fft_samples)
    time = np.arange(fft_samples.size) / sdr.sample_rate

    # update plots
    iq_scatter.setData(i, q)
    iq_plot.setXRange(-1.0, 1.0) # fixes axes from autoscaling every frame
    iq_plot.setYRange(-1.0, 1.0)
    phase_curve.setData(time, phase)

    psd = power_spectral_density(fft_samples)
    curve.setData(freqs, psd)
    waterfall = np.roll(waterfall, 1, axis=0) # push waterfall row down
    waterfall[0, :] = psd # set waterfall row to current data
    image.setImage(waterfall, autoLevels=False)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL) 

capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

def close_devices():
    sdr.cancel_read_async()
    capture_thread.join(timeout=1.0)
    audio_stream.stop()
    audio_stream.close()
    sdr.close()


app.aboutToQuit.connect(close_devices)
QtWidgets.QApplication.instance().exec_()
