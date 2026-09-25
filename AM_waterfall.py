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
RANGE = [88 * 1e6, 108 * 1e6]

sdr.sample_rate = SAMPLE_RATE # Hz
sdr.gain = 20

OUTPUT_RATE = 48_000
DECIMATION = SAMPLE_RATE // OUTPUT_RATE
RF_CUTOFF = 100_000
AUDIO_CUTOFF = 15_000
N_SAMPLES = 2 ** 20
SUBDIVISIONS = 512
CUTOFF = 2

freq_range = np.arange(RANGE[0], RANGE[1], SAMPLE_RATE)
print(freq_range)

radio_stations = {
    "Light FM": 89.9,
    "SYN": 90.7,
    "Smooth 91.5": 91.5,
    "3ZZZ": 92.3,
    "SBS Radio 2": 93.1,
    "Nova 100": 100.3,
    "KIIS 101.1": 101.1,
    "Fox 101.9": 101.9,
    "Triple R": 102.7,
    "Fine Music 103.5": 103.5,
    "Gold 104.3": 104.3,
    "Triple M Melbourne": 105.1,
    "ABC Classic": 105.9,
    "PBS FM": 106.7,
    "ABC triple j": 107.5,
    "Yarra Valley FM": 99.1,
}

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

waterfall = generate_waterfall(freq_range[0])
for central_freq in freq_range[1:]:
    print(f"Generating waterfall for {central_freq / 1e6} MHz")
    current_waterfall = generate_waterfall(central_freq)
    waterfall = np.concatenate([waterfall, current_waterfall], axis=1)

vmin, vmax = np.percentile(waterfall, [2, 98])
plt.imshow(
    waterfall, 
    vmin=vmin, 
    vmax=vmax, 
    cmap='plasma', 
    aspect="auto", 
    origin="lower", 
    extent=[(freq_range[0] - SAMPLE_RATE / 2) / 1e6, (freq_range[-1] + SAMPLE_RATE / 2) / 1e6, 0, (N_SAMPLES / SAMPLE_RATE)])
plt.xlabel("frequency (MHz)")
plt.ylabel("time (s)")
# draw text labels for each radio station
for station, central_freq in radio_stations.items():
    plt.text(central_freq, (N_SAMPLES / SAMPLE_RATE) * 1.025, f"{station}: {central_freq}", color='black', fontsize=8, ha='center', va='bottom', rotation=90)
plt.tight_layout()
plt.show()
sdr.close()

