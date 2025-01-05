import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np


# Set the page configuration
st.set_page_config(
    page_title="Food Recognition",  # Title on the browser tab
    page_icon="🍴",  # Optional: Add an emoji as the icon
    layout="centered",  # Centered layout
    initial_sidebar_state="collapsed"  # Sidebar starts collapsed
)

# Load the trained model
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)

# Define class labels
class_names = ['ayam bakar', 'ayam goreng', 'bakso', 'capcay', 'donat','ikan bakar', 'ikan goreng',
 'kentang goreng', 'kentang rebus', 'nasi',  'puding', 'rendang',
 'roti tawar', 'sate', 'sop', 'tahu goreng', 'telur ceplok',
 'telur dadar', 'telur rebus', 'tempe goreng', 'tumis kangkung']

# Nutrition information
nutrition_info = {
    'puding': "Calories: 150, Protein: 3g, Fat: 5g, Carbs: 25g",
    'kentang goreng': "Calories: 312, Protein: 3.4g, Fat: 15g, Carbs: 41g",
    'bakso': "Calories: 95.0, Protein: 4.7g, Fat: 2.8g, Carbs: 12.8g",
    'capcay': "Calories: 120, Protein: 5g, Fat: 2g, Carbs: 18g",
    'nasi': "Calories: 187.5, Protein: 6.1g, Fat: 4.0g, Carbs: 29.1g",
    'sate': "Calories: 194.5, Protein: 15.4g, Fat: 9.3g, Carbs: 12.1g",
    'telur dadar': "Calories: 200, Protein: 7g, Fat: 15g, Carbs: 2g",
    'sop': "Calories: 48.4, Protein: 5.0g, Fat: 1.8g, Carbs: 3.0g",
    'telur ceplok': "Calories: 92, Protein: 6g, Fat: 7g, Carbs: 1g",
    'roti tawar': "Calories: 70, Protein: 2g, Fat: 1g, Carbs: 13g",
    'rendang': "Calories: 278.5, Protein: 11.4g, Fat: 4.0g, Carbs: 49.1g",
    'ayam goreng': "Calories: 282.2, Protein: 35.3g, Fat: 14.1g, Carbs: 0.8g",
    'tumis kangkung': "Calories: 90, Protein: 3g, Fat: 5g, Carbs: 10g",
    'ayam bakar': "Calories: 220, Protein: 25g, Fat: 10g, Carbs: 3g",
    'tahu goreng': "Calories: 115.0, Protein: 9.7g, Fat: 8.5g, Carbs: 2.5g",
    'donat': "Calories: 250, Protein: 4g, Fat: 12g, Carbs: 30g",
    'ikan bakar': "Calories: 200, Protein: 25g, Fat: 10g, Carbs: 0g",
    'tempe goreng': "Calories: 435.0, Protein: 29.3g, Fat: 32.8g, Carbs: 12.2g",
    'telur rebus': "Calories: 68, Protein: 6g, Fat: 5g, Carbs: 0.5g",
    'ikan goreng': "Calories: 250, Protein: 20g, Fat: 15g, Carbs: 5g",
    'kentang rebus': "Calories: 80, Protein: 2g, Fat: 0.1g, Carbs: 18g"
}


# Preprocessing function
def preprocess_image(uploaded_file):
    """Preprocess the uploaded image to match the model input size."""
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Streamlit app configuration
st.title("Food Recognition App")
st.write("Upload an image of a food item to predict what it is!")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Processing image...")

    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(uploaded_file)

        # Make prediction
        prediksi = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediksi)
        confidence = prediksi[0][predicted_class]

        # Display prediction
        predicted_label = class_names[predicted_class]
        st.markdown(f"""
        ### 🥗 **Prediction Result**
        - **Predicted Class:** 🎯 `{predicted_label}`
        - **Confidence Level:** 🔥 `{confidence:.2f}`

        """)

        # Display nutrition information
        if predicted_label in nutrition_info:
            st.write(f"### 🍴 Nutrition Information (per serving) for **{predicted_label}** 🍴")
            calories, protein, fat, carbs = nutrition_info[predicted_label].split(", ")
            
            # Use Markdown for better styling
            st.markdown(f"""
            | **Nutrient** | **Value** |
            |--------------|-----------|
            | **Calories** | {calories.split(': ')[1]} |
            | **Protein**  | {protein.split(': ')[1]} |
            | **Fat**      | {fat.split(': ')[1]} |
            | **Carbs**    | {carbs.split(': ')[1]} |
            """)


    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please upload an image file to proceed.")