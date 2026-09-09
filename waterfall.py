import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import butter, sosfilt
from scipy.io.wavfile import write
import time

# SDR setup
sdr = RtlSdr()

CENTRAL_FREQ = 100.3 * 1e6 # Hz
SAMPLE_RATE = int(2.4e6)
RANGE = [90 * 1e6, 110 * 1e6]

sdr.sample_rate = SAMPLE_RATE # Hz
sdr.gain = 20

OUTPUT_RATE = 48_000
DECIMATION = SAMPLE_RATE // OUTPUT_RATE
RF_CUTOFF = 100_000
AUDIO_CUTOFF = 15_000
SAMPLING_INTERVAL = 0.1 # seconds
CUTOFF = 0.002 
N_SAMPLES = 2 ** 20
SUBDIVISIONS = 512
CUTOFF = 2
num_samples = int(sdr.sample_rate * SAMPLING_INTERVAL)

range = np.arange(RANGE[0], RANGE[1], SAMPLE_RATE)

def power_spectral_density(samples, sample_rate, n_fft=512):
    # remaining = len(samples) - len(samples)
    blocks = samples.reshape(-1, n_fft)
    waterfall = np.empty((blocks.shape[0], n_fft), dtype=np.float64)

    for i, block in enumerate(blocks):
        spectrum = np.fft.fft(block)
        psd = (np.abs(spectrum) ** 2) / sample_rate
        psd_db = 10 * np.log10(psd + 1e-30)
        waterfall[i, :] = np.fft.fftshift(psd_db)

    return waterfall

def generate_waterfall(central_freq):
    samples = []

    sdr.center_freq = central_freq
    samples = sdr.read_samples(N_SAMPLES)
    waterfall = power_spectral_density(samples, sdr.sample_rate, n_fft=SUBDIVISIONS)
    # freqs = (np.arange(central_freq - SAMPLE_RATE / 2, central_freq + SAMPLE_RATE / 2, SAMPLE_RATE / N_SAMPLES)) / 1e6

    waterfall = waterfall[CUTOFF:, :]
    return waterfall

waterfall = generate_waterfall(range[0])
for freq in range[1:]:
    current_waterfall = generate_waterfall(freq)
    waterfall = np.concatenate([waterfall, current_waterfall], axis=1)

vmin, vmax = np.percentile(waterfall, [2, 98])
plt.imshow(
    waterfall, 
    vmin=vmin, 
    vmax=vmax, 
    cmap='plasma', 
    aspect="auto", 
    origin="lower", 
    extent=[(RANGE[0] - SAMPLE_RATE / 2) / 1e6, (RANGE[1] - SAMPLE_RATE / 2) / 1e6, 0, (N_SAMPLES / SAMPLE_RATE) * len(range)])
plt.xlabel("frequency (MHz)")
plt.ylabel("time (s)")

plt.tight_layout()
plt.show()
sdr.close()

