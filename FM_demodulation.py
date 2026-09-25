import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import butter, sosfilt
from scipy.io.wavfile import write

# SDR setup
sdr = RtlSdr()

SAMPLE_RATE = int(1.2e6)
sdr.sample_rate = SAMPLE_RATE # Hz
sdr.center_freq = 100.3 * 1e6 # Hz
sdr.gain = "auto"

OUTPUT_RATE = 48_000
DECIMATION = SAMPLE_RATE // OUTPUT_RATE
RF_CUTOFF = 100_000
AUDIO_CUTOFF = 15_000
SAMPLING_INTERVAL = 10.0 # seconds
CUTOFF = 0.002 
num_samples = int(sdr.sample_rate * SAMPLING_INTERVAL)

target = int(SAMPLE_RATE * SAMPLING_INTERVAL)
samples = []
remaining = target
chunk_size = 256000

while remaining > 0:
    n = min(chunk_size, remaining)
    block = sdr.read_samples(n)
    samples.append(block)
    remaining -= len(block)

samples = np.concatenate(samples)

# Filter the RF channel before demodulation and downsampling.
rf_filter = butter(5, RF_CUTOFF, btype="lowpass", fs=SAMPLE_RATE, output="sos")
filtered_samples = sosfilt(rf_filter, samples)

i = np.real(filtered_samples)
q = np.imag(filtered_samples)
time = np.arange(filtered_samples.size) / sdr.sample_rate

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.scatter(i, q, s=1, alpha=0.3)
ax1.set_xlabel("I")
ax1.set_ylabel("Q")
ax1.set_title("Q vs I (complex)")
ax1.set_aspect("equal", adjustable="box")
ax1.grid(True)

phase_differences = np.angle(
    filtered_samples[1:] * np.conj(filtered_samples[:-1]) # naturally gives wrapped smallest phase
)

# Remove high-frequency demodulator noise before downsampling.
audio_filter = butter(
    5, AUDIO_CUTOFF, btype="lowpass", fs=SAMPLE_RATE, output="sos"
)
phase_differences = sosfilt(audio_filter, phase_differences)
time = time[1:]

# throw away first 2 milliseconds
mask = time > CUTOFF
time = time[mask]
phase_differences = phase_differences[mask]

max_val = np.max(np.abs(phase_differences))
if max_val > 0:
    normalized_data = phase_differences / max_val
else:
    normalized_data = phase_differences



# Convert to 16-bit PCM integers (-32768 to 32767)
audio_data = (normalized_data * 32767).astype(np.int16)
audio_samples = normalized_data[::DECIMATION]
audio_data = (audio_samples * 32767).astype(np.int16)
write("output.wav", OUTPUT_RATE, audio_data)

ax2.plot(time, phase_differences)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase Differences (rad)")
ax2.set_title("Phase differences vs time")
ax2.grid(True)

plt.tight_layout()
plt.show()
sdr.close()
