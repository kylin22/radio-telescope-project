import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from rtlsdr import RtlSdr

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
UPDATE_INTERVAL = 50

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

# plotting the waterfall
waterfall_plot = pg.PlotWidget(labels={"left": "Time", "bottom": "Frequency (MHz)"})
waterfall_plot.setMouseEnabled(x=False, y=False)
waterfall_plot.invertY(True)
image = pg.ImageItem(waterfall, axisOrder="row-major")  
waterfall_plot.addItem(image)

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


def update():
    global waterfall

    samples = sdr.read_samples(FFT_SIZE)
    # calculate data for IQ and phase plots
    i = np.real(samples)
    q = np.imag(samples)
    phase = np.angle(samples)
    time = np.arange(samples.size) / sdr.sample_rate

    # update plots
    iq_scatter.setData(i, q)
    iq_plot.setXRange(-1.0, 1.0) # fixes axes from autoscaling every frame
    iq_plot.setYRange(-1.0, 1.0)
    phase_curve.setData(time, phase)

    psd = power_spectral_density(samples)
    curve.setData(freqs, psd)
    waterfall = np.roll(waterfall, 1, axis=0) # push waterfall row down
    waterfall[0, :] = psd # set waterfall row to current data
    image.setImage(waterfall, autoLevels=False)



timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL) 

app.aboutToQuit.connect(sdr.close)
QtWidgets.QApplication.instance().exec_()
