import streamlit as st
from streamlit_app import run_app

def main():
    st.title("Face Recognition App")
    st.write("Upload a target image and a group image to identify faces.")

    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    group_image = st.file_uploader("Upload Group Image", type=["jpg", "jpeg", "png"])

    if st.button("Identify Faces"):
        if target_image is not None and group_image is not None:
            run_app(target_image, group_image)
        else:
            st.error("Please upload both images.")

if __name__ == "__main__":
    main()
    