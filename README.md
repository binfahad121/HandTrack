# HandTrack — Real-Time Hand Tracking & Gesture Control

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%200.10.9+-00C8FF.svg)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-macOS%20(Apple%20Silicon%20%7C%20Intel)-lightgrey.svg)](https://apple.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A portfolio-quality, modular computer-vision desktop application built for macOS. HandTrack processes live camera frames on-device using Google's **MediaPipe Tasks Hand Landmarker**, rendering a futuristic cybernetic heads-up display (HUD) in OpenCV with real-time gesture recognition, air drawing, and safe virtual mouse interaction.

---

## Key Features

- **Real-Time 3D Hand Tracking**:
  - Multi-hand tracking (up to 2 hands) with all 21 3D landmarks.
  - Natural mirror-view selfie display.
  - Cybernetic skeleton renderer with glowing fingertip halos and handedness confidence badges.
- **Scale-Invariant Gesture Recognition**:
  - Robust classification invariant to hand distance from camera:
    - ✋ **Open Palm** (Navigation / Idle)
    - ✊ **Fist** (Standby)
    - ☝️ **Index Pointing** (AirDraw brush trigger & cursor pointer)
    - ✌️ **Two Fingers / Victory** (Gesture mode recognition)
    - 🤏 **Thumb-Index Pinch** (Left click & drag trigger)
  - Temporal hysteresis voting buffer eliminates frame jitter and gesture flickering.
- **Air Drawing Digital Canvas**:
  - Natural finger-in-the-air sketching triggered by pointing.
  - 5 selectable neon palette colors (Neon Cyan, Electric Green, Solar Yellow, Hot Pink, Flame Orange).
  - Eraser mode with visual circular eraser reticle.
  - Undo stack (`Z`) to step back strokes.
  - Clear canvas (`C`) and instant high-resolution PNG export (`S`) saved to `saved_drawings/`.
  - Disconnect filtering prevents accidental connecting lines when hand moves rapidly.
- **Safety-First Virtual Mouse**:
  - Exponential Moving Average (EMA) cursor smoothing eliminates hand trembling.
  - Central interaction box mapping provides relaxed physical hand reach without straining to frame edges.
  - Edge-triggered pinch click: clicks once upon pinch contact, preventing runaway click loops.
  - Sustained pinch-and-hold triggers mouse drag (`mouseDown` / `mouseUp`).
  - Strict safety switch: mouse control is **disabled by default** and requires explicit activation (`M`).
  - PyAutoGUI failsafe protection enabled.
- **Futuristic OpenCV HUD Interface**:
  - Dark glass theme with rounded panels, cyan and electric green accents.
  - Smoothed live FPS counter and real-time telemetry card.
  - Active boundary guidelines, gesture badges, and animated transient toast notifications.
  - In-app interactive help modal (`H`).
- **100% Local & Private**:
  - All vision algorithms execute locally on your machine.
  - Zero video streaming, cloud uploads, or telemetry collection.

---

## Architecture & Codebase Structure

HandTrack is engineered with strict separation of concerns across dedicated modules:

```
HandTrack/
├── main.py                  # Application entry point, camera lifecycle & event loop
├── hand_tracker.py          # MediaPipe Tasks Hand Landmarker wrapper & skeleton renderer
├── gesture_controller.py    # Scale-invariant geometric gesture classifier & filter
├── mouse_controller.py      # Virtual cursor mapping, EMA smoothing, and click debouncing
├── drawing_canvas.py        # Multi-stroke air canvas, eraser, undo stack & PNG exporter
├── ui.py                    # OpenCV cybernetic HUD: headers, telemetry cards & toast notifications
├── config.py                # Central dataclasses, palette definitions & application modes
├── requirements.txt         # Pinned core dependencies
├── pytest.ini               # Pytest path and runner configuration
├── models/
│   └── hand_landmarker.task # Official Google MediaPipe Tasks model asset (~7.5 MB)
├── saved_drawings/          # Default directory for exported PNG drawings
└── tests/                   # Automated unit and integration test suite
    ├── test_config.py       # Configuration and reset tests
    ├── test_gestures.py     # Deterministic gesture tests with synthetic landmarks
    ├── test_canvas.py       # Stroke drawing, undo, clear, and PNG export tests
    ├── test_mouse.py        # Screen mapping, smoothing math, and safety tests
    └── test_tracker.py      # Missing model error handling and lifecycle tests
```

---

## Operating Modes

| Mode | Shortcut | Description |
| :--- | :---: | :--- |
| **Mode 1: Live Tracking** | `1` | Full 21-landmark skeleton visualization, normalized 3D coordinates telemetry, and handedness tracking. |
| **Mode 2: Virtual Mouse** | `2` | Control macOS cursor with your index finger. Press `M` to arm mouse control. Pinch to click or drag. |
| **Mode 3: Air Draw** | `3` | Point index finger to sketch in neon light. Press `E` for eraser, `Z` to undo, `C` to clear, `S` to export PNG. |
| **Mode 4: Gesture Recognition** | `4` | Real-time classification with live breakdown of individual finger flexion/extension states. |

---

## Keyboard Controls Cheat Sheet

| Key | Action |
| :---: | :--- |
| `1` | Switch to **Live Tracking** mode |
| `2` | Switch to **Virtual Mouse** mode |
| `3` | Switch to **Air Draw** mode |
| `4` | Switch to **Gesture Recognition** mode |
| `M` | Toggle **Virtual Mouse control ON / OFF** (Safety gate) |
| `C` | **Clear** drawing canvas |
| `Z` | **Undo** most recent drawing stroke |
| `E` | Toggle **Eraser** mode |
| `S` | **Save** current drawing as PNG to `saved_drawings/` |
| `[` | Decrease cursor sensitivity by 0.1x |
| `]` | Increase cursor sensitivity by 0.1x |
| `R` | Reset all settings back to factory defaults |
| `H` | Toggle interactive full-screen **Help Guide** modal |
| `Q` or `ESC` | Safely release webcam and **Quit** application |

---

## Installation & Quickstart

### Prerequisites

- macOS 12.0+ (Tested on Apple Silicon `arm64` and Intel `x86_64`)
- Python 3.9, 3.10, or 3.11

### 1. Clone & Set Up Virtual Environment

```bash
cd HandTrack

# Create dedicated virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Upgrade pip and install pinned packages without precompiling tests
python -m pip install --upgrade pip
python -m pip install --no-compile -r requirements.txt
```

### 3. Model Download & Verification

The application automatically verifies and downloads the official Google MediaPipe `hand_landmarker.task` file (~7.5 MB) on first launch. If you wish to pre-fetch it manually:

```bash
mkdir -p models
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 4. Run Automated Tests

Verify that your local environment and algorithm logic are completely sound:

```bash
# Run 20 unit tests across all controllers
.venv/bin/pytest -v

# Run automated headless pipeline smoke test (synthetic frames end-to-end)
.venv/bin/python main.py --headless-test
```

### 5. Launch HandTrack

```bash
.venv/bin/python main.py
```

Optional CLI flags:
```bash
# Specify camera index (default: 0) and resolution
.venv/bin/python main.py --camera 0 --width 1280 --height 720
```

---

## macOS Permissions Guide

Because HandTrack accesses the webcam and controls OS mouse events, macOS requires granting explicit permissions to your terminal:

### 1. Camera Access
1. Open **System Settings** → **Privacy & Security** → **Camera**.
2. Ensure your terminal app (**Terminal**, **iTerm2**, or **Visual Studio Code**) is toggled **ON**.
3. If prompted with *"Terminal would like to access the camera"*, click **Allow**.

### 2. Accessibility Access (For Virtual Mouse)
1. Open **System Settings** → **Privacy & Security** → **Accessibility**.
2. Add and toggle **ON** your terminal app (**Terminal**, **iTerm2**, or **VS Code**).
3. This grants permission for PyAutoGUI to safely reposition the cursor and register clicks.
> *Note: Virtual mouse control is completely optional and remains inactive until you press `M`.*

---

## User Interface Preview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ (•) HANDTRACK          [ MODE 3: AIR DRAW ]                  FPS: 60 (•)   │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────┐                                                 │
│ │ TELEMETRY & SENSORS     │                                                 │
│ │ ─────────────────────── │                                                 │
│ │ Hands: 1 ACTIVE         │                                                 │
│ │ Gesture:                │                                                 │
│ │ ┌─────────────────────┐ │                                                 │
│ │ │ ☝️ Index Pointing    │ │                 *  *                            │
│ │ └─────────────────────┘ │               *  (8)  *   <-- Fingertip pointer │
│ │ Tool: Neon Cyan         │                 *  *                            │
│ │ Palette: (o) [Cyan]     │                  │                              │
│ │ Drawn Strokes: 3        │                 (7)                             │
│ └─────────────────────────┘                  │                              │
│                                              ...                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ [1-4] Modes | [M] Mouse | [C] Clear | [Z] Undo | [E] Eraser | [H] Help ...  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

- **Error: `OpenCV: not authorized to capture video`**:
  - Cause: macOS camera permission missing for your terminal emulator.
  - Solution: Follow the [macOS Permissions Guide](#macos-permissions-guide) above, then restart your terminal.
- **Error: `PyAutoGUI failSafeException`**:
  - Cause: Moving mouse cursor directly to the top-left corner triggers PyAutoGUI's built-in safety tripwire.
  - Solution: HandTrack automatically catches this exception and safely releases pressed buttons without crashing. Press `M` to re-enable when ready.
- **Slow First Import**:
  - Cause: Matplotlib builds its system font cache on macOS during the first invocation. HandTrack stores this in `.mplconfig/` so subsequent launches are instantaneous.

---

## Privacy Statement

HandTrack is designed for privacy:
- Webcam frames are captured directly into RAM and processed in-memory using local CPU/GPU acceleration.
- No video feeds, biometric data, or analytics are saved or transmitted across any network.
- Drawings are saved exclusively to your local `saved_drawings/` folder only when you explicitly press `S`.

---

## Future Roadmap

- [ ] Multi-hand collaborative air drawing (left hand selects color wheel, right hand sketches).
- [ ] Custom dynamic gesture recorder (record custom finger sequences for custom macro actions).
- [ ] Air keyboard overlay with pinch typing.
- [ ] Virtual volume and brightness knob control.
