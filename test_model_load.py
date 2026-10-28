from pathlib import Path
from tensorflow.keras.models import load_model

MODEL_PATH = Path("model") / "best_model_fixed.keras"

print(f"Loading model from: {MODEL_PATH.absolute()}")

if not MODEL_PATH.exists():
    print(f"FILE NOT FOUND: {MODEL_PATH.absolute()}")
else:
    model = load_model(str(MODEL_PATH), compile=False)
    print("")
    print("MODEL LOADED SUCCESSFULLY")
    print("")
    model.summary()
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
