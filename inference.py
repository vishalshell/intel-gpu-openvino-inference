from openvino.runtime import Core
import numpy as np

# Load the OpenVINO model
core = Core()
model = core.read_model("model/model.xml")
compiled_model = core.compile_model(model, "GPU")  # Force GPU usage

# Get input layer
input_layer = compiled_model.input(0)
input_shape = input_layer.shape

# Create dummy input
dummy_input = np.random.rand(*input_shape).astype(np.float32)

# Run inference
result = compiled_model([dummy_input])

# Display result
print("Inference result shape:", result[compiled_model.output(0)].shape)
print("Success: Model ran on Intel Integrated GPU using OpenVINO.")
