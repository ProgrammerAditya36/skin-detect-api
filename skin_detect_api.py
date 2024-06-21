from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
model_path = 'model_checkpoint.weights.keras'
model = load_model(model_path)
app = FastAPI()
class SkinDetectRequest(BaseModel):
    data: list
@app.post("/predict")
def predict_skin_detect(request: SkinDetectRequest):
    data = np.array(request.data)
    predictions = model.predict(data)
    class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    result = f'Predicted class: {predicted_class_name} with probability {predictions[0][predicted_class_index]:.4f}'
    return {"result": result}
@app.get("/")
def read_root():
    return {"Hello": "World"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)