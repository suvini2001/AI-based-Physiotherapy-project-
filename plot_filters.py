"""plot_filters.py
Plot Raw vs Top-Performing Filter Candidates on real elbow exercise data.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filters import (butterworth_filter, savitzky_golay_filter, 
                     rolling_median_filter, OneEuroFilter)

# 1. සැබෑ කෝණ දත්ත කියවීම (Person A ගේ වීඩියෝවෙන් හැදූ CSV එක)
df = pd.read_csv("data/elbow_A_angles.csv")
time = df["time_s"].values
raw_angles = df["elbow_angle"].values
FS = 29.5  # සැබෑ frame rate එක

# 2. Comparison එකෙන් තේරුණු හොඳම Filters යෙදීම:
# (A) පළමුව Spikes අයින් කිරීමට Median filter එක
clean_spikes = rolling_median_filter(raw_angles, window_size=5)

# (B) Offline Winner 1: Median + Savitzky-Golay (9, 3)
savgol_angles = savitzky_golay_filter(clean_spikes, window_length=9, polyorder=3)

# (C) Offline Winner 2: Median + Butterworth 6 Hz (filtfilt)
butter_angles = butterworth_filter(clean_spikes, fs=FS, cutoff=6.0, causal=False)

# (D) Live Winner: Causal Butterworth 6 Hz (lfilter with lfilter_zi)
butter_live_angles = butterworth_filter(raw_angles, fs=FS, cutoff=6.0, causal=True)

# 3. ප්‍රස්තාරය ඇඳීම
plt.figure(figsize=(12, 6))

plt.plot(time, raw_angles, label="Raw Angle (No Filter)", color="gray", alpha=0.5, linewidth=1.0)
plt.plot(time, savgol_angles, label="Offline Best: Median 5 + Savitzky-Golay (9,3)", color="green", linewidth=1.8)
plt.plot(time, butter_angles, label="Offline Best: Median 5 + Butterworth 6Hz", color="blue", linestyle="--", linewidth=1.8)
plt.plot(time, butter_live_angles, label="Live Candidate: Butterworth 6Hz (Causal)", color="red", linewidth=1.5)

plt.xlabel("Time (seconds)", fontsize=11)
plt.ylabel("Elbow Angle (degrees)", fontsize=11)
plt.title("Elbow Angle Smoothing: Comparison of Top Filter Candidates", fontsize=13, fontweight="bold")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":", alpha=0.6)

# 4. රූපයක් ලෙස save කිරීම (Deliverable එක)
plt.tight_layout()
plt.savefig("data/filter_comparison_elbow.png", dpi=300)
plt.show()

print("Comparison plot saved successfully to data/filter_comparison_elbow.png")