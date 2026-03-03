\# 🖱️ Virtual Mouse using Hand Gesture



> Control your computer mouse completely hands-free using real-time hand gesture recognition powered by \*\*MediaPipe\*\* and \*\*OpenCV\*\* — no physical mouse needed.



---



\## 🎯 Features



| Gesture | Action |

|---------|--------|

| ☝️ Move Index Finger | Move mouse cursor |

| 👌 Thumb + Index close | Single Click |

| 🤏 Thumb + Middle close | Double Click |

| ✊ Thumb + Index (hold) | Drag and Drop |

| 🤙 Thumb + Pinky close | Decrease Volume |

| 🤙 Thumb + Pinky medium | Increase Volume |

| 🖐️ Thumb + Ring close | Decrease Brightness |

| 🖐️ Thumb + Ring medium | Increase Brightness |

| ✌️ Index + Middle close (up) | Scroll Up |

| ✌️ Index + Middle close (down) | Scroll Down |



---



\## 🛠️ Tech Stack



| Library | Purpose |

|---------|---------|

| OpenCV | Camera feed and frame processing |

| MediaPipe | Real-time hand landmark detection |

| PyAutoGUI | Mouse movement and click control |

| PyCaw | System volume control |

| Screen Brightness Control | Monitor brightness control |

| NumPy | Distance calculations between landmarks |



---



\## 📁 Project Structure



```

Virtual-Mouse-using-Hand-Gesture/

├── Virtual Mouse using Hand Gesture.ipynb    # Main notebook

└── README.md                                 # Documentation

```



---



\## 🚀 Quick Start



\### 1. Clone the repository

```bash

git clone https://github.com/Onkar-1203/Virtual-Mouse-using-Hand-Gesture.git

cd Virtual-Mouse-using-Hand-Gesture

```



\### 2. Install dependencies

```bash

pip install opencv-python mediapipe pyautogui pycaw screen-brightness-control numpy comtypes

```



\### 3. Run the notebook

Open `Virtual Mouse using Hand Gesture.ipynb` in Jupyter Notebook and run all cells.



\### 4. Stop the program

Press \*\*`Q`\*\* on your keyboard to quit.



---



\## ⚙️ How It Works



\### Step 1 — Camera Feed

OpenCV captures live video from your webcam and flips it horizontally for a mirror effect.



\### Step 2 — Hand Detection

MediaPipe detects 21 hand landmarks in real time on every frame.



\### Step 3 — Landmark Mapping

Key fingertip positions are extracted:

\- \*\*Landmark 4\*\* → Thumb tip

\- \*\*Landmark 8\*\* → Index finger tip

\- \*\*Landmark 12\*\* → Middle finger tip

\- \*\*Landmark 16\*\* → Ring finger tip

\- \*\*Landmark 20\*\* → Pinky tip



\### Step 4 — Gesture Recognition

Distances between fingertips are calculated using \*\*Euclidean distance\*\*. Each gesture is triggered when the distance crosses a specific threshold.



\### Step 5 — Smooth Cursor Movement

A \*\*smoothing factor (0.2)\*\* is applied to cursor movement to prevent jitter and make motion fluid.



\### Step 6 — System Control

PyAutoGUI controls the mouse. PyCaw controls volume. Screen Brightness Control adjusts display brightness.



---



\## 📊 Gesture Distance Thresholds



| Action | Fingers | Threshold |

|--------|---------|-----------|

| Single Click | Thumb + Index | < 30px |

| Drag | Thumb + Index | < 40px |

| Double Click | Thumb + Middle | < 35px |

| Scroll | Index + Middle | < 35px |

| Volume Down | Thumb + Pinky | < 50px |

| Volume Up | Thumb + Pinky | 50–100px |

| Brightness Down | Thumb + Ring | < 30px |

| Brightness Up | Thumb + Ring | 30–80px |



---



\## ⚠️ Requirements



\- Python 3.8+

\- Webcam

\- Windows OS (PyCaw is Windows only)

\- Good lighting for accurate hand detection



---



\## 🔮 Future Improvements



\- \[ ] Right click support

\- \[ ] Multi-hand gesture support

\- \[ ] Custom gesture mapping

\- \[ ] MacOS / Linux support

\- \[ ] GUI for gesture configuration



---



\## 👤 Author



\*\*Onkar\*\* — \[github.com/Onkar-1203](https://github.com/Onkar-1203)



---



\## 📝 License



MIT License — free to use for portfolio and commercial projects.

