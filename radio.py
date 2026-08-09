import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from rtlsdr import RtlSdr

# SDR setup
sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 100e6
sdr.gain = "auto"

# Options
FFT_SIZE = 8192 # the frequency resolution is sample_rate / FFT_SIZE
WATERFALL_ROWS = 120 # how much history to keep on screen
POWER_MIN = -50.0 # should be around the noise floor (min estimate of noise power in dB)
POWER_MAX = 50
UPDATE_INTERVAL = 50

window = np.hanning(FFT_SIZE) # filter out signal leakage across bins
freqs_raw = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / sdr.sample_rate) + sdr.center_freq)
freqs = freqs_raw / 1e6 # MHz 

# setup
app = QtWidgets.QApplication([])
win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout(win)

# plotting the spectrum
waterfall = np.full((WATERFALL_ROWS, FFT_SIZE), -50.0)
spectrum_plot = pg.PlotWidget(labels={"left": "Power (dB)", "bottom": "Frequency (MHz)"})
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
image.setLevels((-30, 20)) # needs to come after colorbar is created for some reason
colorbar.orientation = "vertical"
colorbar.setFixedWidth(120)
waterfall_row.addWidget(colorbar)
layout.addLayout(waterfall_row)

# window setup
win.setWindowTitle("RTL-SDR Spectrum")
win.showMaximized()

def get_power_spectral_density():
    samples = sdr.read_samples(FFT_SIZE) # list of complex IQ samples
    spectrum = np.fft.fftshift(np.fft.fft(samples * window)) # run FFT with hanning
    dB = 20 * np.log10(np.abs(spectrum) + 1e-12) # converts to dB (need +1e-12 to prevent log(0))
    return dB

def update():
    global waterfall
    psd = get_power_spectral_density()
    curve.setData(freqs, psd)
    waterfall = np.roll(waterfall, 1, axis=0) # push waterfall row down
    waterfall[0, :] = psd # set waterfall row to current data
    image.setImage(waterfall, autoLevels=False)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(UPDATE_INTERVAL) 

app.aboutToQuit.connect(sdr.close)
QtWidgets.QApplication.instance().exec_()
