# Elephant Behavior Classification Prototype

Multimodal research prototype for **Elephant Behavior Classification** (Normal vs Aggressive) for railway safety in Sri Lanka.

## Project Structure

- `config.py`
- `main.py`
- `core/`
- `pose/`
- `sound/`
- `io_modules/`
- `utils/`
- `models/` (placeholders for model files)

## PC Test Flow (No Raspberry Pi required)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run one cycle with webcam + optional microphone:
   ```bash
   python main.py --once
   ```
3. Run with sample audio file:
   ```bash
   python main.py --once --sample-audio path/to/sample.wav
   ```
4. Run with video input instead of webcam:
   ```bash
   python main.py --once --camera-source path/to/video.mp4
   ```

The pipeline will:
- Try pose inference from camera/video frame.
- Try sound inference from sample wav or microphone.
- Fuse predictions.
- Build LoRa payload.
- Print payload when serial LoRa hardware is unavailable.

## Model File Placeholders

Place trained models in `models/`:
- `pose_rf.pkl`
- `pose_scaler.pkl`
- `pose_top_features.json`
- `sound_rf.pkl`
- `sound_xgb.pkl`
- `sound_cnn_lstm.h5`
- `sound_scaler.pkl`
- `sound_hybrid_scaler.pkl`
- `label_encoder.pkl`

## Raspberry Pi Readiness

Use the **same** `main.py` and folder structure. Only adjust configuration/CLI:
- camera source
- serial port
- thresholds/weights
- test mode

If any Pi hardware/libraries are missing, modules degrade gracefully and continue with available modalities.
