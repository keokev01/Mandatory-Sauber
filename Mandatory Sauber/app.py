import gradio as gr
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('C:\Users\Kevin Keo\OneDrive - ZHAW\Dokumente\Desktop\ZHAW WIN\FS24\KI-Anwendung\Pokemon\Mandatory Sauber\Mandatory Sauber\mein_modell.h5')  # Ensure this path is correct

def predict_pokemon(img):
    img = Image.fromarray(img.astype('uint8'), 'RGB')  # Ensure the image is in RGB
    img = img.resize((224, 224))  # Resize the image properly using PIL
    img_array = keras_image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to fit model input
    img_array = preprocess_input(img_array)  # Preprocess the input as expected by ResNet50
    
    prediction = model.predict(img_array)  # Predict using the model
    classes = ['Butterfree', 'Cubone', 'Dratini' ]  # Specific Pokémon names
    return {classes[i]: float(prediction[0][i]) for i in range(3)}  # Return the prediction

# Define Gradio interface
interface = gr.Interface(fn=predict_pokemon, 
                         inputs="image",  # Simplified input type
                         outputs="label",  # Simplified output type
                         title="Pokémon Classifier",
                         description="Upload an image of a Pokémon and the classifier will predict its species.")

# Launch the interface
interface.launch()
