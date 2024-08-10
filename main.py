import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model #type: ignore

# Load your Keras model
cancer_model = load_model("skin_cancer1.keras")
try: 
    detect_model = load_model("skin_disease1.keras")
except:

    detect_model = load_model("skin_cancer1.keras")

# FastAPI application
app = FastAPI()

# Pydantic model for request data
class SkinDetectRequest(BaseModel):
    data: list
# Endpoint for POST requests to predict skin detection
@app.post("/detect")
def detect_skin(request: SkinDetectRequest):
    data = np.array(request.data)
    if data.size == 0:
        raise HTTPException(status_code=400, detail="Empty data array")
    
    predictions = detect_model.predict(data)
    class_names = ['Enfeksiyonel', 'Ekzema','Acne','Pigment','Benign','Malign']
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    probability = float(predictions[0][predicted_class_index])
    return {"result": predicted_class_name, "probability": probability}
@app.post("/predict")
def predict_skin_detect(request: SkinDetectRequest):
    data = np.array(request.data)
    if data.size == 0:
        raise HTTPException(status_code=400, detail="Empty data array")
    
    predictions = cancer_model.predict(data)
    class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
                   'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    probability = float(predictions[0][predicted_class_index])  # Convert to float
    
    return {"result": predicted_class_name, "probability": probability}

# Root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Endpoint for GET requests to /predict (should not be used for predictions)
@app.get("/predict")
def predict():
    return {"error": "Please make a POST request to /predict endpoint with data"}
@app.get("/detect")
def detect():
    return {"error": "Please make a POST request to /detect endpoint with data"}
# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
