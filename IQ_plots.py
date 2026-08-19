import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.io.wavfile import write

# SDR setup
sdr = RtlSdr()

SAMPLE_RATE = int(1.2e6)
sdr.sample_rate = SAMPLE_RATE # Hz
sdr.center_freq = 100.3 * 1e6 # Hz
sdr.gain = "auto"

SAMPLING_INTERVAL = 2.0 # seconds
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

i = np.real(samples)
q = np.imag(samples)
phase = np.angle(samples)
time = np.arange(num_samples) / sdr.sample_rate

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
ax1.scatter(i, q, s=1, alpha=0.3)
ax1.set_xlabel("I")
ax1.set_ylabel("Q")
ax1.set_title("Q vs I (complex)")
ax1.set_aspect("equal", adjustable="box")
ax1.grid(True)

phase_differences = np.unwrap(phase)
phase_differences = np.diff(phase_differences)
time = time[0:-1]

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
write("output.wav", SAMPLE_RATE, audio_data)

ax2.plot(time, phase_differences)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase Differences (rad)")
ax2.set_title("Phase differences vs time")
ax2.grid(True)

plt.tight_layout()
plt.show()
sdr.close()
