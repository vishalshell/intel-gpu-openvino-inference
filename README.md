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

Vishal — [UST Global](https://ust.com)

## 📄 License

MIT License


---

## 🧪 Full Walkthrough: MobileNetV2 Inference on Intel iGPU

This example uses the [MobileNetV2](https://github.com/onnx/models/tree/main/vision/classification/mobilenet) model from the ONNX Model Zoo to demonstrate how to:

1. Download a pretrained model
2. Convert it to OpenVINO format
3. Run inference using Intel Integrated GPU (iGPU)

---

### 📥 Step 1: Download Pretrained Model (ONNX)

```bash
wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx -O mobilenetv2.onnx
```

---

### 🔄 Step 2: Convert ONNX to OpenVINO IR Format

```bash
mo --input_model mobilenetv2.onnx --output_dir model/ --data_type FP16
```

This creates:
```
model/
├── mobilenetv2.xml
└── mobilenetv2.bin
```

---

### 🧠 Step 3: Run Inference Using Python

Save the following as `mobilenet_infer.py`:

```python
import numpy as np
import cv2
from openvino.runtime import Core

# Load the model on Intel GPU
core = Core()
model_path = "model/mobilenetv2.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "GPU")

# Load and preprocess the image
img = cv2.imread("cat.jpg")
if img is None:
    img = np.random.rand(224, 224, 3) * 255  # fallback image
img = cv2.resize(img, (224, 224))
img = img.transpose(2, 0, 1).astype(np.float32)  # CHW
img = img[np.newaxis, :] / 255.0

# Run inference
output = compiled_model([img])
result = list(output.values())[0]
top_class = np.argmax(result)

# Optional: Load ImageNet labels
try:
    import json
    with open("imagenet-simple-labels.json") as f:
        labels = json.load(f)
    print("Predicted:", labels[top_class])
except:
    print("Top predicted class ID:", top_class)
```

Download label file (optional):

```bash
wget https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json
```

---

### ✅ Run It

```bash
python mobilenet_infer.py
```

If successful, the model runs entirely on your **Intel iGPU** and outputs the top predicted class.

---

### 📎 Credits

- MobileNetV2 ONNX model: [ONNX Model Zoo](https://github.com/onnx/models)
- ImageNet Labels: [Anish Athalye](https://github.com/anishathalye/imagenet-simple-labels)
- OpenVINO Toolkit: [Intel](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)

