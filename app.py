import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

model = load_model("pneumonia_model.h5")
st.title("Medical Diagnosis of Pneumonia")
uploaded=st.file_uploader("Upload a chest X-ray image", type=["png","jpeg","jpg","AVIF"])
if uploaded is not None:
    img=Image.open(uploaded)
    st.image(img, caption="Uploaded Image for Prediction", use_container_width=True)
    if st.button("Predict"):
        img=img.resize((64,64))
        if img.mode!='RGB':
            img=img.convert('RGB')
        img_array=np.array(img)/255.0
        img_array=np.expand_dims(img_array,axis=0)
        prediction=model.predict(img_array)
        confidence=prediction[0][0]
        if confidence>0.5:
            label="Pneumonia"
            confidence_per=confidence*100
        else:
            label="Normal"
            confidence_per=(1-confidence)*100
        st.write(f"Prediction: **{label}** with confidence **{confidence_per:.2f}%**")
