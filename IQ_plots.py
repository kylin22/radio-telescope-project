import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr

# SDR setup
sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 100e6
sdr.gain = "auto"

SAMPLING_INTERVAL = 0.01 # seconds
num_samples = int(sdr.sample_rate * SAMPLING_INTERVAL)
samples = sdr.read_samples(num_samples)

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

ax2.plot(time, phase, lw=0.5)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Phase (rad)")
ax2.set_title("Phase vs time")
ax2.grid(True)

plt.tight_layout()
plt.show()
sdr.close()
