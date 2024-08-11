import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model #type: ignore

# Load your Keras models
cancer_model = load_model("skin_cancer1.keras")
detect_model = load_model("skindisease.keras")

# FastAPI application
app = FastAPI()

# Pydantic model for request data
class SkinDetectRequest(BaseModel):
    data: list

class ChatRequest(BaseModel):
    message: str

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

# New endpoint for chat
@app.post("/chat")
def chat(request: ChatRequest):
    user_message = request.message.lower()
    
    # Simple dummy responses
    if "hello" in user_message or "hi" in user_message:
        return {"response": "Hello! How can I assist you with skin health today?"}
    elif "skin" in user_message:
        return {"response": "Skin health is important. Make sure to protect your skin from the sun and stay hydrated!"}
    elif "cancer" in user_message:
        return {"response": "If you're concerned about skin cancer, it's best to consult with a dermatologist for a professional evaluation."}
    else:
        return {"response": "I'm here to help with skin-related questions. Could you please provide more details about your concern?"}

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