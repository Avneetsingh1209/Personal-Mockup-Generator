import streamlit as st
from PIL import Image
import numpy as np
import zipfile
import io
import cv2
import os
import tempfile
import atexit
import shutil

st.set_page_config(page_title="Shirt Mockup Generator", layout="centered")
st.title("👕 Shirt Mockup Generator with Batching")

st.markdown("""
Upload multiple design PNGs and shirt templates.  
Preview placement and generate mockups in batches.
""")

# --- Sidebar Controls ---
plain_padding_ratio = st.sidebar.slider("Padding Ratio – Plain Shirt", 0.1, 1.0, 0.45, 0.05)
model_padding_ratio = st.sidebar.slider("Padding Ratio – Model Shirt", 0.1, 1.0, 0.45, 0.05)
plain_offset_pct = st.sidebar.slider("Vertical Offset – Plain Shirt (%)", -50, 100, 24, 1)
model_offset_pct = st.sidebar.slider("Vertical Offset – Model Shirt (%)", -50, 100, 38, 1)

# --- Disk Space Check ---
total, used, free = shutil.disk_usage("/")
free_gb = free // (2**30)
total_gb = total // (2**30)

st.sidebar.markdown("### 💾 Disk Space Info")
st.sidebar.write(f"Free: **{free_gb} GB** / Total: {total_gb} GB")

if free_gb < 2:  # less than 2 GB left
    st.sidebar.error("⚠️ Low disk space! Large mockups may fail.")

# --- Session Setup ---
if "zip_files_output" not in st.session_state:
    st.session_state.zip_files_output = {}
if "design_names" not in st.session_state:
    st.session_state.design_names = {}

# --- Upload Section ---
design_files = st.file_uploader("📌 Upload Design Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
shirt_files = st.file_uploader("🎨 Upload Shirt Templates", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Clear Button ---
if st.button("🔄 Start Over (Clear Generated Mockups)"):

    # cleanup temp file if exists
    if "generated_zip" in st.session_state:
        if os.path.exists(st.session_state.generated_zip):
            os.remove(st.session_state.generated_zip)

    for key in ["design_files", "design_names", "zip_files_output", "generated_zip"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Design Naming ---
if design_files:
    st.markdown("### ✏️ Name Each Design")
    for i, file in enumerate(design_files):
        default_name = os.path.splitext(file.name)[0]
        custom_name = st.text_input(
            f"Name for Design {i+1} ({file.name})", 
            value=st.session_state.design_names.get(file.name, default_name),
            key=f"name_input_{i}_{file.name}"
        )
        st.session_state.design_names[file.name] = custom_name

# --- Batch Controls ---
if design_files:
    st.markdown("### 📦 Batch Processing Control")
    total_designs = len(design_files)
    batch_start = st.number_input("Start from Design #", min_value=1, max_value=total_designs, value=1)
    batch_end = st.number_input("End at Design #", min_value=batch_start, max_value=total_designs, value=min(batch_start + 19, total_designs))
    selected_batch = design_files[batch_start - 1: batch_end]
else:
    selected_batch = []

# --- Estimated Output Size ---
enough_space = True
if design_files and shirt_files:
    estimated_size_per_mockup = 0.8 * 1024 * 1024  # ~0.8 MB per PNG
    num_designs = len(selected_batch) if selected_batch else len(design_files)
    num_shirts = len(shirt_files)
    estimated_total_size = num_designs * num_shirts * estimated_size_per_mockup

    estimated_mb = int(estimated_total_size / (1024 * 1024))
    st.sidebar.markdown("### 📦 Estimated Output Size")
    st.sidebar.write(f"~{estimated_mb} MB (for {num_designs} designs × {num_shirts} shirts)")

    if estimated_total_size > free:
        st.sidebar.error("❌ Not enough disk space for this batch!")
        enough_space = False
    elif estimated_total_size > free * 0.8:
        st.sidebar.warning("⚠️ This batch may nearly fill your disk!")

# --- Bounding Box Detection ---
def get_shirt_bbox(pil_image):
    img_cv = np.array(pil_image.convert("RGB"))[:, :, ::-1]
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        return cv2.boundingRect(largest)
    return None

# --- Live Preview ---
if design_files and shirt_files:
    st.markdown("### 👀 Live Preview")
    selected_design = st.selectbox("Select a Design", design_files, format_func=lambda x: x.name)
    selected_shirt = st.selectbox("Select a Shirt Template", shirt_files, format_func=lambda x: x.name)

    try:
        selected_design.seek(0)
        design = Image.open(selected_design).convert("RGBA")
        selected_shirt.seek(0)
        shirt = Image.open(selected_shirt).convert("RGBA")

        is_model = "model" in selected_shirt.name.lower()
        offset_pct = model_offset_pct if is_model else plain_offset_pct
        padding_ratio = model_padding_ratio if is_model else plain_padding_ratio

        bbox = get_shirt_bbox(shirt)
        if bbox:
            sx, sy, sw, sh = bbox
            scale = min(sw / design.width, sh / design.height, 1.0) * padding_ratio
            new_width = int(design.width * scale)
            new_height = int(design.height * scale)
            resized_design = design.resize((new_width, new_height))
            y_offset = int(sh * offset_pct / 100)
            x = sx + (sw - new_width) // 2
            y = sy + y_offset
        else:
            resized_design = design
            x = (shirt.width - design.width) // 2
            y = (shirt.height - design.height) // 2

        preview = shirt.copy()
        preview.paste(resized_design, (x, y), resized_design)
        st.image(preview, caption="📸 Live Mockup Preview", use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Preview failed: {e}")

# --- Cleanup helper ---
def safe_delete(path):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception as e:
            print(f"⚠️ Could not delete temp file {path}: {e}")

# Register cleanup on exit
def cleanup_on_exit():
    if "generated_zip" in st.session_state:
        safe_delete(st.session_state.generated_zip)

atexit.register(cleanup_on_exit)

# --- Generate Mockups ---
if st.button("🚀 Generate Mockups for Selected Batch", disabled=not enough_space):
    if not (selected_batch and shirt_files):
        st.warning("Upload at least one design and one shirt template.")
    else:
        # cleanup old file if exists
        if "generated_zip" in st.session_state:
            safe_delete(st.session_state.generated_zip)

        # Create new zip on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmpfile:
            with zipfile.ZipFile(tmpfile, "w", zipfile.ZIP_DEFLATED) as master_zipf:
                for design_file in selected_batch:
                    graphic_name = st.session_state.design_names.get(design_file.name, "graphic")
                    design_file.seek(0)
                    design = Image.open(design_file).convert("RGBA")

                    inner_zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(inner_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                        for shirt_file in shirt_files:
                            color_name = os.path.splitext(shirt_file.name)[0]
                            shirt_file.seek(0)
                            shirt = Image.open(shirt_file).convert("RGBA")

                            is_model = "model" in shirt_file.name.lower()
                            offset_pct = model_offset_pct if is_model else plain_offset_pct
                            padding_ratio = model_padding_ratio if is_model else plain_padding_ratio

                            bbox = get_shirt_bbox(shirt)
                            if bbox:
                                sx, sy, sw, sh = bbox
                                scale = min(sw / design.width, sh / design.height, 1.0) * padding_ratio
                                new_width = int(design.width * scale)
                                new_height = int(design.height * scale)
                                resized_design = design.resize((new_width, new_height))
                                y_offset = int(sh * offset_pct / 100)
                                x = sx + (sw - new_width) // 2
                                y = sy + y_offset
                            else:
                                resized_design = design
                                x = (shirt.width - design.width) // 2
                                y = (shirt.height - design.height) // 2

                            shirt_copy = shirt.copy()
                            shirt_copy.paste(resized_design, (x, y), resized_design)

                            output_name = f"{graphic_name}_{color_name}_tee.png"
                            img_byte_arr = io.BytesIO()
                            shirt_copy.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)
                            zipf.writestr(output_name, img_byte_arr.getvalue())

                    inner_zip_buffer.seek(0)
                    master_zipf.writestr(f"{graphic_name}.zip", inner_zip_buffer.read())

            tmpfile_path = tmpfile.name
            st.session_state.generated_zip = tmpfile_path  # store path

# --- Download Button ---
if "generated_zip" in st.session_state:
    file_path = st.session_state.generated_zip
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            st.download_button(
                label="📦 Download All Mockups (Grouped by Design)",
                data=f,
                file_name="all_mockups_by_design.zip",
                mime="application/zip",
                key="download_zip"
            )
