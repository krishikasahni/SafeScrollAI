# 🛡️ SafeScrollAI

> **AI-Powered Violence Detection in Videos using Deep Learning**

SafeScrollAI is a deep learning-based web application that automatically detects violent content in uploaded videos. The system analyzes video frames and classifies videos into different risk categories while providing confidence scores, AI recommendations, visual analytics, downloadable reports, and detection history.

---

## 📌 Project Overview

Online platforms receive thousands of videos every minute, making manual moderation difficult and time-consuming. SafeScrollAI assists content moderators by automatically analyzing uploaded videos and identifying potentially violent content.

The application provides:

- 🎥 Video upload and preview
- 🧠 AI-powered violence detection
- 📊 Confidence score visualization
- 🚨 Risk level classification
- 🕒 Video risk timeline
- 🖼️ Sample frame gallery
- 🤖 AI-generated recommendations
- 📜 Detection history
- 📈 Interactive analytics dashboard
- 📄 PDF report generation
- 📥 CSV export of detection history

---

# ✨ Features

### 🎥 Video Upload

- Upload MP4, AVI and MOV videos
- Automatic video preview
- Video metadata extraction

---

### 🧠 AI Prediction

The uploaded video is processed using a trained deep learning model that predicts whether the content contains violence.

Prediction Categories:

- ✅ Safe Video
- ⚠️ Possible Violence
- 🚨 Violence Detected

---

### 📊 Confidence Score

Displays the prediction confidence using an interactive Plotly gauge.

---

### 🤖 AI Recommendation

Automatically suggests moderation actions based on the predicted risk level.

Example recommendations:

- Human moderation
- Age restriction
- Viewer discretion warning
- Safe for public viewing

---

### 🕒 Video Risk Timeline

Displays a color-coded timeline representing predicted violence intensity throughout the video.

Legend:

🟢 Safe

🟡 Possible Violence

🔴 Violence

---

### 🖼️ Sample Frame Gallery

Displays representative frames extracted from different parts of the uploaded video.

---

### 📜 Detection History

Stores previous analyses locally and displays:

- Video name
- Prediction
- Risk level
- Confidence
- Date
- Time

Users can:

- Download history as CSV
- Clear previous history

---

### 📄 PDF Report

Generates a professional analysis report containing:

- Video information
- Prediction
- Confidence
- Risk level
- Processing time
- AI recommendation

---

### 📈 Analytics Dashboard

Interactive charts include:

- Risk Distribution Pie Chart
- Confidence Trend
- Overall Detection Statistics
- Average Confidence

---