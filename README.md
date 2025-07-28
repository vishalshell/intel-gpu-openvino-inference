# Intel iGPU AI Inference with OpenVINO

This project demonstrates how to run AI inference on an **Intel Integrated GPU (iGPU)** using the **OpenVINO Toolkit**. It forces execution on GPU for optimized edge inference without relying on CUDA.

## 🚀 Features

- ✅ Run inference exclusively on Intel iGPU
- ✅ Use OpenVINO for model loading and execution
- ✅ Simple, self-contained example with dummy input
- ✅ Plug-and-play for any IR (.xml/.bin) model

## 📦 Requirements

- Python 3.8+
- Intel iGPU (UHD, Iris Xe, Arc, etc.)
- OpenVINO Toolkit
- Converted IR model (`model.xml` and `model.bin`)

## 🔧 Setup

1. **Install OpenVINO**
```bash
pip install openvino
```

2. **Convert your model to OpenVINO IR (if needed)**
```bash
mo --input_model model.onnx
```

3. **Place the `model.xml` and `model.bin` inside the `model/` directory.**

## ▶️ Run Inference

```bash
python inference.py
```

## 📂 Directory Layout

```
.
├── inference.py
├── model/
│   ├── model.xml
│   └── model.bin
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧪 Notes

- This example is **inference-only**, not training.
- TensorFlow, PyTorch, and Stable Diffusion are **not supported** on Intel iGPU.
- For training or heavy AI tasks, use NVIDIA GPU or cloud GPU (Colab, AWS, etc.).

## 🧠 Related Links

- [OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)
- [Model Optimizer Guide](https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)

## 👨‍💻 Author

Vishal (vishal.im@gmail.com)

## 📄 License

MIT License
