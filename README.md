# 🤟 SignLang AI Vision — Real-Time Sign Language Translator

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
[![CI](https://github.com/deswanth12/singlangbydeshu/actions/workflows/ci.yml/badge.svg)](https://github.com/deswanth12/singlangbydeshu/actions/workflows/ci.yml)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Computer%20Vision-orange.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)
![Responsive](https://img.shields.io/badge/Responsive-Mobile%20%26%20PC-brightgreen.svg)

A high-performance, real-time **AI Sign Language Translator** web dashboard and desktop application powered by **MediaPipe Computer Vision**, **Flask**, and **Web Speech Synthesis**.

Tracks 21 3D landmarks per hand across **single and dual-hand gestures**, translates static & conversational signs into text, constructs sentences, synthesizes spoken audio in multiple languages, and features a gamified sign language training quiz game!

---

## 🌟 Key Features

- **👐 Real-Time Dual-Hand Landmark Tracking**: Detects and tracks 21 3D nodes per hand simultaneously using webcam input with custom glowing neon visualization overlays.
- **📐 Scale & Rotation Invariant Extension Metrics**: Uses scale-invariant finger knuckle-to-tip ratios ($\frac{\|\text{Tip} - \text{Wrist}\|}{\|\text{Knuckle} - \text{Wrist}\|}$) for high-accuracy gesture detection under any camera angle or distance.
- **💬 Live Sentence Accumulator**: Accumulates stable hand signs into words and sentences with individual word chip deletion, debouncing, and space management.
- **🔊 Multi-Language Text-to-Speech Engine**: Translates and speaks out loud in **English, Spanish, French, German, Hindi, and Japanese** with customizable speech rates.
- **🎮 Gamified Sign Language Quiz Game**: Challenge yourself with a 10-second countdown game, live target sign matching, streak counters, and score tracking.
- **🎙️ Reverse Speech-to-Sign Visualizer**: Speak out loud into your microphone to view animated sign language flashcards for each word spoken.
- **🔴 Custom Gesture Recorder**: Record custom hand landmark positions, name them, and save them to `custom_signs.json` dynamically.
- **📊 Real-Time Dual-Hand Finger Diagnostic Meters**: Live HUD meters showing individual finger extension levels for Left Hand (Cyan) and Right Hand (Pink).
- **🎨 Dynamic Canvas Skeleton Themes**: Switch between **⚡ Cyberpunk Neon**, **🟢 Matrix Emerald**, **🌈 Rainbow Flow**, and **✨ Minimal Nodes**.
- **🔄 Mobile & PC Responsive**: Includes mobile touch targets, responsive breakpoints, and **🔄 Flip Cam** switching for front selfie vs rear mobile cameras.
- **💾 One-Click Export**: Download sentences as `.txt` files or share directly to WhatsApp.

---

## 📸 Supported Gestures

| Category | Gestures Included |
| --- | --- |
| **Conversational** | `Hello` (Open Palm), `Yes` (Fist), `Good` (Thumbs Up), `Bad` (Thumbs Down), `Peace / V` (Victory), `I Love You`, `OK`, `You` (Pointing), `Rock`, `Call Me`, `Hold / Stop`, `Small` (Pinch) |
| **ASL Alphabet** | Letters A, B, C, D, E, I, L, V, W, Y |
| **ASL Digits** | Digits 0, 1, 2, 3, 4, 5 |
| **Dual-Hand Signs** | `Thank You / Please` (Praying Hands), `No / Cancel` (Crossed Wrists), `Love / Heart` (Heart Hands), `Awesome` (Double Thumbs Up), `Book / Read` (Open Book), `Home` (Roof), `Applause` (Clapping), `Double Victory`, `Welcome` |
| **Custom Signs** | Dynamically user-recorded gestures saved in app |

---

## 📦 Project Structure

```
singlang/
├── app.py                   # Flask Web API server & gesture endpoints
├── gestures.py              # Geometric vector extraction & fuzzy matching classifier
├── utils.py                 # Mathematical metrics, extension ratios, & TemporalSmoother
├── main.py                  # Standalone OpenCV Desktop Application
├── requirements.txt         # Dependencies (opencv-python, mediapipe, numpy, pyttsx3, flask)
├── static/
│   ├── css/
│   │   └── style.css        # Futuristic dark glassmorphism stylesheet
│   └── js/
│       └── main.js          # Client-side MediaPipe runner, quiz engine, & speech synthesizer
├── templates/
│   └── index.html           # Responsive web dashboard HUD
├── LICENSE                  # MIT License
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

---

## ⚡ Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/singlang.git
cd singlang

# (Optional) Create virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Run Web Application

```bash
python app.py
```
Open your browser to:
👉 **`http://127.0.0.1:5000`**

### 3. Run Standalone Desktop Application

```bash
python main.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Function |
| --- | --- |
| <kbd>Space</kbd> | Insert space into sentence accumulator |
| <kbd>Backspace</kbd> | Delete last word |
| <kbd>S</kbd> | Speak full sentence out loud |
| <kbd>C</kbd> | Clear sentence buffer |
| <kbd>T</kbd> | Toggle Text-to-Speech (Desktop app) |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
