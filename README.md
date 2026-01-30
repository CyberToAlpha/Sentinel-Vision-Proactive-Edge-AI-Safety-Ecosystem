# 🛡️ Sentinel Vision: Proactive Edge-AI Safety Ecosystem
> "Moving L&T Safety from Reactive Monitoring to Predictive Prevention."


## 🚀 Key Features
- **Edge-Native:** Runs offline on NVIDIA Jetson / RTX 4050 (<45ms latency).
- **Privacy-First:** Auto-blurring of faces compliant with DPDP Act 2026.
- **Drone-Ready:** Compatible with aerial inspection feeds.
- **Robust:** Hard Negative Mining to ignore "Yellow Buckets" vs Helmets.


## 🛠️ Tech Stack
- **AI Engine:** YOLOv8 (Custom Trained) + TensorRT (FP16 Optimization)
- **Tracking:** ByteTrack (Occlusion Handling)
- **Interface:** Streamlit
- **Hardware:** Optimized for CUDA 11.8+

## ⚡ How to Run
```bash
git clone [https://github.com/CyberToAlpha/sentinel-vision.git](https://github.com/CyberToAlpha/sentinel-vision.git)
pip install -r requirements.txt
python src/app.py
