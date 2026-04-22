import streamlit as st
import numpy as np
import cv2
import pandas as pd

from models import load_model, predict_large_image
from postprocess import detect_craters

st.set_page_config(layout="wide")
st.title("🌌 Planetary Crater Detection System")

# ======================
# SESSION STATE INIT
# ======================
for key in ["prob_map", "mask", "overlay", "annotated", "df", "last_file"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ======================
# INPUT
# ======================
uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

model_name = st.selectbox(
    "Select Model",
    ["Ghost-RDT-UNet++", "RDT-UNet++"]
)

# ======================
# RESET ON NEW IMAGE
# ======================
if uploaded_file:
    if st.session_state.last_file != uploaded_file.name:
        st.session_state.last_file = uploaded_file.name
        for key in ["prob_map", "mask", "overlay", "annotated", "df"]:
            st.session_state[key] = None

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image")

    # ======================
    # SEGMENTATION
    # ======================
    if st.button("Run Segmentation"):
        with st.spinner("Running segmentation..."):
            model = load_model(model_name)
            prob_map = predict_large_image(model, image)

        mask = (prob_map > 0.5).astype(np.uint8) * 255

        overlay = image.copy()
        overlay[mask == 255] = (0,255,0)

        st.session_state.prob_map = prob_map
        st.session_state.mask = mask
        st.session_state.overlay = overlay

    # ======================
    # SHOW SEGMENTATION
    # ======================
    if st.session_state.mask is not None:
        st.subheader("Segmentation Results")

        col1, col2 = st.columns(2)

        with col1:
            st.image(st.session_state.mask, caption="Mask")

        with col2:
            st.image(cv2.cvtColor(st.session_state.overlay, cv2.COLOR_BGR2RGB),
                     caption="Overlay")

        # downloads
        _, buf1 = cv2.imencode(".png", st.session_state.mask)
        st.download_button("Download Mask", buf1.tobytes(), "mask.png")

        _, buf2 = cv2.imencode(".png", st.session_state.overlay)
        st.download_button("Download Overlay", buf2.tobytes(), "overlay.png")

    # ======================
    # POST PROCESSING
    # ======================
    if st.button("Post Processing"):
        if st.session_state.prob_map is None:
            st.warning("Run segmentation first")
        else:
            craters = detect_craters(st.session_state.prob_map)

            annotated = image.copy()
            data = []

            for i,(x,y,r) in enumerate(craters):
                cv2.circle(annotated, (int(x),int(y)), int(r), (0,0,255), 2)
                data.append([i,x,y,r])

            df = pd.DataFrame(data, columns=["id","x","y","radius"])

            st.session_state.annotated = annotated
            st.session_state.df = df

    # ======================
    # SHOW POST PROCESSING
    # ======================
    if st.session_state.annotated is not None:
        st.subheader("Post Processing Results")

        st.image(cv2.cvtColor(st.session_state.annotated, cv2.COLOR_BGR2RGB))

        st.dataframe(st.session_state.df)

        _, buf = cv2.imencode(".png", st.session_state.annotated)
        st.download_button("Download Annotated Image", buf.tobytes(), "annotated.png")

        st.download_button("Download CSV",
                           st.session_state.df.to_csv(index=False).encode(),
                           "craters.csv")