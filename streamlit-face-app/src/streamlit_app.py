import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import os

from face_utils import segregate_group_image

APP_DIR = Path(__file__).parent
SAVED_GROUPS_DIR = APP_DIR / "saved_groups"
SAVED_GROUPS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    st.title("Clicks")
    st.write("Upload a target image and one or more group images to identify faces. Group photos that contain the target will be saved.")

    target_image = st.file_uploader("Upload Target Image", type=["jpg", "jpeg", "png"])
    group_images = st.file_uploader("Upload Group Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Process Images"):
        if target_image is None:
            st.error("Please upload a target image.")
            return
        if not group_images:
            st.error("Please upload at least one group image.")
            return

        # load target as RGB numpy
        target_image_pil = Image.open(target_image).convert("RGB")
        target_image_np = np.array(target_image_pil)

        processed_info = []
        saved_paths = []

        for gfile in group_images:
            # read group image as RGB numpy
            g_pil = Image.open(gfile).convert("RGB")
            g_np = np.array(g_pil)

            st.write(f"Processing: {gfile.name} ...")
            summary = segregate_group_image(target_image_np, g_np)

            # display visualization (summary["visualization"] is RGB numpy)
            vis = summary.get("visualization")
            if vis is not None:
                st.image(vis, caption=f"Visualization: {gfile.name}", use_column_width=True)
            else:
                # fallback to showing original
                st.image(g_np, caption=f"{gfile.name} (no visualization)", use_column_width=True)

            # if matches found, save the original group photo to saved_groups and record path
            if summary.get("matches"):
                safe_name = f"{Path(gfile.name).stem}_{len(summary['matches'])}match{Path(gfile.name).suffix}"
                out_path = SAVED_GROUPS_DIR / safe_name
                # convert RGB->BGR for cv2.imwrite
                out_bgr = cv2.cvtColor(g_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), out_bgr)
                saved_paths.append(str(out_path))
                st.success(f"Target found in {gfile.name} — saved to: {out_path}")
            else:
                st.info(f"No target found in {gfile.name}")

            processed_info.append({"filename": gfile.name, "summary": summary})

        # show summary list of saved group photos
        if saved_paths:
            st.write("Saved group images containing the target:")
            for p in saved_paths:
                st.write(p)
        else:
            st.write("No group images contained the target.")

        # show small gallery of all processed group images
        st.write("All processed group images:")
        cols = st.columns(3)
        for i, info in enumerate(processed_info):
            img_np = np.array(Image.open(group_images[i]).convert("RGB"))
            with cols[i % 3]:
                st.image(img_np, caption=info["filename"], use_column_width=True)

if __name__ == "__main__":
    main()