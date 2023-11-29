# Metal Classification Model



## Description

The Defect Detection App is a web application that utilizes machine learning models to predict whether the top part of a submersible pump impeller is defective or not. The models are trained on images of the top part of a submersible pump impeller using a pre-trained ResNet model for feature extraction and a custom defect detection model for classification. The app has been deployed on Streamlite.

Note: Deployed version of the web pages [Here](https:// /)

## Notebooks and dataset

- [Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

- [Classification model](https://colab.research.google.com/drive/1iNE_vqm_ldkyepU1-zqdXWsXoKWYkkuN#scrollTo=-6OFw-eLwLP7)

- [Image validation](https://colab.research.google.com/drive/1ymbm-72w7LZ7VScm497O1j8nXTO6AcK8#scrollTo=l1oNCvHDID6ehttps%3A%2F%2Fcolab.research.google.com%2Fdrive%2F1ymbm-72w7LZ7VScm497O1j8nXTO6AcK8%23scrollTo%3Dl1oNCvHDID6e)

## Features
- Image Upload: Users can upload an image of the top part of a submersible pump impeller for defect detection.
- Relevance Check: The app checks if the uploaded image is relevant by comparing its features to a pre-defined clustering model.
- Prediction: After uploading the image, users can click the "Predict" button to get the model's prediction regarding the defect status.

## Packages Used

This project has used the some packages such as numpy, tensorflow, which have to be installed to run this web app locally present in `requirements.txt` file. 

## Installation

To run the project locally, there is a need to have Visual Studio Code (vs code) installed on your PC:

- **[vs code](https://code.visualstudio.com/download)**: It is a source-code editor made by Microsoft with the Electron Framework, for Windows, Linux, and macOS.

## Usage

1. Clone the project 

``` bash
git clone https://github.com/UmuhireJessie/metal-classification.git

```

2. Open the project with vs code

``` bash
cd metal-classification
code .
```

3. Install the required dependencies

``` bash
pip install -r requirements.txt
```


4. Run the project

``` bash
streamlit run app.py
```

5. Use the link printed in the terminal to visualise the app. (Usually `http://127.0.0.1:5000/`)

## Model Files

- defect_detection_model.h5: The main defect detection model trained on top parts of submersible pump impeller images.
- image_validation_model.h5: A model used to validate if the uploaded image is relevant to the defect detection task.
- scaler.pkl: The scaler used for standardizing features during inference.
- kmeans_model.pkl: The KMeans clustering model for checking the relevance of the uploaded image.
- train_features.pkl: Features extracted from the training set for clustering.

## Important Notes
- The app is designed to work specifically with images of the top part of a submersible pump impeller.
- Images that do not belong to this category will be rejected.

## Authors and Acknowledgment

- Jessie Umuhire Umutesi
- Adrine Uwera
- Evelyne Umubyeyi
- Gabin Ishimwe

## License
[MIT](https://choosealicense.com/licenses/mit/)
