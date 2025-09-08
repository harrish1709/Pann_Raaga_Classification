# 🎵 Pann Raaga Comparison Web App

A Flask-based web application for **analyzing and comparing two Carnatic music recordings**.  
It estimates tonics, extracts swaras (notes), compares note distributions, melodic n-grams, and pitch contours, and visualizes the results.

---

## ✨ Features

- 🎶 **Automatic tonic estimation** using `librosa.pyin`.
- 🎵 **Swara mapping (12-tone)** with octave awareness.
- 📊 **Duration-weighted note distributions** and bigram/trigram phrase analysis.
- 📈 **Pitch contour visualization**.
- 🔗 **Dynamic Time Warping (DTW)** for melodic contour similarity.
- 🌐 Web-based upload interface powered by Flask.
