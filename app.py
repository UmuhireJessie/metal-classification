import streamlit as st 
import numpy as np
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model("model/defect_detection_model.h5")

image_validation_model = tf.keras.models.load_model("model/image_validation_model.h5")

# Load the scaler from the file
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the kmeans model from the file
with open('model/kmeans_model.pkl', 'rb') as kmeans_file:
    kmeans = pickle.load(kmeans_file)

# Load the kmeans model from the file
with open('model/train_features.pkl', 'rb') as train_features_file:
    train_features = pickle.load(train_features_file)


st.set_page_config(
    page_title="Defect detection App",
    page_icon="images/metal_icon.jpeg"
)

st.title("⚙️ Defect detection App ⚙️")

st.markdown("<div style='padding: 30px'></div>", unsafe_allow_html=True)
st.subheader("Welcome to Our Trained Defect Detection Model")
st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)
st.markdown("""Test if your top part of a submersible pump impeller is defected and prevent costly effects later! Upload the image of the top part of a submersible pump impeller and
                    get the response from the model""")
st.markdown("""
    The model utilizes the predetrained model; Resnet model, to perform feature
    extraction and uses the features extracted to build it. While the user can upload any image to
    the model but we have limited the model to accept only the top part of a submersible pump impeller images since it was only
    trained on  the top part of a submersible pump impeller images. To generate predictions, the users must upload the image, and click the
    submit button.
      
""")

st.markdown("<div style='padding: 50px'></div>", unsafe_allow_html=True)
st.subheader("Upload the image: ")
st.markdown("<div style='padding: 10px'></div>", unsafe_allow_html=True)

img = st.file_uploader("Insert the image")
st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)

if(img):
    st.image(img)

st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)

predict = st.button("Predict")

st.markdown("<div style='padding: 20px'></div>", unsafe_allow_html=True)

# Convert the features to a 2D array
train_features_2d = train_features.reshape(train_features.shape[0], -1)

train_features_std = scaler.fit_transform(train_features_2d)

# Predict the cluster labels for the training set
train_cluster_labels = kmeans.predict(train_features_std)

# Function to check if an image is relevant based on clustering
def is_relevant_by_cluster(image, model, kmeans, scaler):
    # Preprocess the image for ResNet50
    preprocessed_image = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(image, axis=0))

    # Extract features using ResNet50
    image_features = model.predict(preprocessed_image)

    # Convert the features to a 2D array
    image_features_2d = image_features.reshape(1, -1)

    # Standardize features
    image_features_std = scaler.transform(image_features_2d)

    # Predict the cluster label
    cluster_label = kmeans.predict(image_features_std)[0]

    return cluster_label == np.argmax(np.bincount(train_cluster_labels))

if predict:
    # Load and preprocess the image for prediction
    image = tf.keras.preprocessing.image.load_img(img)
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array_2 = tf.keras.preprocessing.image.img_to_array(image)
    # Expand dimensions to create batch size of 1
    image_array = np.expand_dims(image_array, axis=0)  
    # Normalize pixel values
    image_array = image_array / 255.0  
    
    if is_relevant_by_cluster(image_array_2, image_validation_model, kmeans, scaler):

        prediction = model.predict(image_array)
 
        if prediction[0] > 0.5:
            st.subheader("The metal is not defected.")
            st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)
        else:
            st.subheader(" The metal is defected. ")
            st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)


    else: 
        st.markdown("<h3 style='color: red; font-weight: bold;'>Image rejected.</h3>", unsafe_allow_html=True)
        st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True)
        st.subheader("Image is not the top part of a submersible pump impeller.")
        st.markdown("<div style='padding: 5px'></div>", unsafe_allow_html=True) 

