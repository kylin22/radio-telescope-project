import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from rtlsdr import RtlSdr

sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 100e6 # FM range
sdr.gain = "auto"

FFT_SIZE = 8192 # the frequency resolution is sample_rate / FFT_SIZE
WATERFALL_ROWS = 120  # how much history to keep on screen

window = np.hanning(FFT_SIZE) # filter out signal leakage across bins
freqs_raw = (np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / sdr.sample_rate)) + sdr.center_freq) 
freqs = freqs_raw / 1e6 # MHz 

def get_power_spectral_density():
    samples = sdr.read_samples(FFT_SIZE) # list of complex IQ samples
    spectrum = np.fft.fftshift(np.fft.fft(samples * window)) # run FFT with hanning
    dB = 20 * np.log10(np.abs(spectrum) + 1e-12) # converts to dB (need +1e-12 to prevent log(0))
    return dB

POWER_MIN = -50.0 # should be around the noise floor (min estimate of noise power in dB)
POWER_MAX = 50
waterfall = np.full((WATERFALL_ROWS, FFT_SIZE), POWER_MIN) 

fig, (ax_spectrum, ax_waterfall) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 2]})
spectrum_plot, = ax_spectrum.plot(freqs, waterfall[-1], lw=0.5)
ax_spectrum.set_xlim(freqs[0], freqs[-1])
ax_spectrum.set_ylim(POWER_MIN, POWER_MAX)
ax_spectrum.set_ylabel("Power (dB)")
ax_spectrum.set_title(f"Center {sdr.center_freq/1e6:.3f} MHz \n Span {sdr.sample_rate/1e6:.2f} MHz")

waterfall_img = ax_waterfall.imshow(
    waterfall, aspect="auto", origin="upper",
    extent=[freqs[0], freqs[-1], 0, WATERFALL_ROWS],
    vmin=POWER_MIN, vmax=POWER_MAX, cmap="plasma",
)
ax_waterfall.invert_yaxis()
ax_waterfall.set_xlabel("Frequency (MHz)")
ax_waterfall.set_ylabel("Time (frames ago)")

def update(_frame):
    global waterfall
    psd = get_power_spectral_density()
    spectrum_plot.set_ydata(psd)
    waterfall = np.roll(waterfall, -1, axis=0) # shift line down each frame
    waterfall[-1, :] = psd
    waterfall_img.set_data(waterfall)
    return spectrum_plot, waterfall_img

UPDATE_INTERVAL = 50
animation = animation.FuncAnimation(fig, update, interval=UPDATE_INTERVAL, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()

sdr.close()
