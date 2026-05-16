import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Fashion Recommendation System",
    page_icon="✨",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(to bottom right, #f8fafc, #eef2ff);
}

/* Main Heading */

.main-title {
    text-align: center;
    font-size: 58px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 20px;
    color: #6b7280;
    margin-bottom: 40px;
}

/* Upload Box */

div[data-testid="stFileUploader"] {
    background: white;
    padding: 25px;
    border-radius: 20px;
    border: 2px dashed #a78bfa;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.06);
}

/* Images */

.stImage img {
    border-radius: 18px;
    transition: 0.3s;
}

.stImage img:hover {
    transform: scale(1.03);
}

/* Recommendation Heading */

.recommendation-heading {
    font-size: 32px;
    font-weight: 700;
    color: #111827;
    margin-top: 20px;
    margin-bottom: 20px;
}

/* Success Message */

.success-box {
    background: #ecfdf5;
    padding: 15px;
    border-radius: 12px;
    color: #065f46;
    font-weight: 600;
    margin-top: 15px;
}

/* Remove Streamlit top spacing */

.block-container {
    padding-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD EMBEDDINGS ----------------

feature_list = np.array(
    pickle.load(open('embeddings.pkl', 'rb'))
)

filenames = pickle.load(
    open('filenames.pkl', 'rb')
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False

    model = tensorflow.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])

    return model

model = load_model()

# ---------------- NEIGHBOR MODEL ----------------

neighbors = NearestNeighbors(
    n_neighbors=6,
    algorithm='auto',
    metric='euclidean'
)

neighbors.fit(feature_list)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">Fashion Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">YOUR FASHION BESTIE!</div>',
    unsafe_allow_html=True
)

# ---------------- SAVE FILE FUNCTION ----------------

def save_uploaded_file(uploaded_file):

    try:

        os.makedirs('uploads', exist_ok=True)

        save_path = os.path.join(
            'uploads',
            uploaded_file.name
        )

        with open(save_path, 'wb') as f:

            f.write(uploaded_file.getbuffer())

        return save_path

    except Exception as e:

        st.error(f"Upload Error: {e}")

        return None

# ---------------- FEATURE EXTRACTION ----------------

def feature_extraction(img_path, model):

    img = image.load_img(
        img_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    expanded_img_array = np.expand_dims(
        img_array,
        axis=0
    )

    preprocessed_img = preprocess_input(
        expanded_img_array
    )

    result = model.predict(
        preprocessed_img,
        verbose=0
    ).flatten()

    normalized_result = result / norm(result)

    return normalized_result

# ---------------- RECOMMEND FUNCTION ----------------

def recommend(features):

    distances, indices = neighbors.kneighbors([features])

    return indices

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=['png', 'jpg', 'jpeg']
)

# ---------------- MAIN APP ----------------

if uploaded_file is not None:

    saved_path = save_uploaded_file(uploaded_file)

    if saved_path:

        col1, col2 = st.columns([1, 2])

        with col1:

            st.subheader("Uploaded Image")

            display_image = Image.open(uploaded_file)

            st.image(
                display_image,
                use_container_width=True
            )

        with st.spinner("Analyzing image and finding recommendations..."):

            features = feature_extraction(
                saved_path,
                model
            )

            indices = recommend(features)

        st.markdown(
            '<div class="success-box">Recommendations generated successfully ✨</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="recommendation-heading">Recommended Products</div>',
            unsafe_allow_html=True
        )

        cols = st.columns(5)

        recommendation_count = 0

        for i in range(1, len(indices[0])):

            try:

                image_path = str(filenames[indices[0][i]])

                # windows path ko linux compatible banayega
                image_path = image_path.replace("\\", "/")

                # sirf filename lega
                image_name = os.path.basename(image_path)

                # final repo image path
                final_path = os.path.join("images", image_name)

                # image exists check
                if os.path.exists(final_path):

                    with cols[recommendation_count]:

                        st.image(
                            final_path,
                            use_container_width=True
                        )

                    recommendation_count += 1

                if recommendation_count == 5:
                    break

            except Exception as e:

                st.warning(f"Image Load Error: {e}")

    else:

        st.error("Error uploading image.")