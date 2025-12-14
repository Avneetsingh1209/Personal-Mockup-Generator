import streamlit as st
from PIL import Image, ImageChops
import numpy as np
import cv2
import zipfile
import io
import os

# ---------------- PAGE ----------------
st.set_page_config(page_title="Realistic T-Shirt Mockup Generator", layout="centered")
st.title("👕 Realistic T-Shirt Mockup Generator")
st.caption("Auto fabric folds • Live preview • Batch export")

# ---------------- SIDEBAR CONTROLS ----------------
padding_ratio = st.sidebar.slider("Design Size", 0.1, 1.0, 0.45, 0.05)
vertical_offset = st.sidebar.slider("Vertical Offset (%)", -50, 100, 30, 1)
horizontal_offset = st.sidebar.slider("Horizontal Offset (%)", -50, 50, 0, 1)
displacement_strength = st.sidebar.slider("Fabric Fold Strength", 0, 50, 20, 1)

# ---------------- UPLOAD ----------------
design_files = st.file_uploader(
    "📌 Upload Design PNGs",
    type=["png"],
    accept_multiple_files=True
)

shirt_files = st.file_uploader(
    "🎨 Upload T-Shirt Images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ---------------- FUNCTIONS ----------------
def get_shirt_bbox(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.boundingRect(max(contours, key=cv2.contourArea))
    return None


def generate_displacement_map(shirt_image):
    img = np.array(shirt_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Laplacian(gray, cv2.CV_32F)
    edges = np.absolute(edges)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(edges, (21, 21), 0)
    disp = cv2.normalize(blurred, None, 100, 155, cv2.NORM_MINMAX)

    return Image.fromarray(disp, "L")


def apply_displacement(design, disp_map, strength):
    design_np = np.array(design)
    disp = np.array(disp_map)

    h, w = disp.shape
    dx = (disp / 255.0 - 0.5) * strength
    dy = dx

    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)

    warped = cv2.remap(
        design_np,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_TRANSPARENT
    )

    return Image.fromarray(warped, "RGBA")


def apply_shadows(design, shirt_crop):
    shadows = shirt_crop.convert("L").resize(design.size)
    shadows = Image.merge("RGBA", (shadows, shadows, shadows, shadows))
    return ImageChops.multiply(design, shadows)

# ---------------- LIVE PREVIEW ----------------
if design_files and shirt_files:
    st.markdown("### 👀 Live Preview")

    selected_design = st.selectbox(
        "Select Design",
        design_files,
        format_func=lambda x: x.name
    )

    selected_shirt = st.selectbox(
        "Select T-Shirt",
        shirt_files,
        format_func=lambda x: x.name
    )

    design = Image.open(selected_design).convert("RGBA")
    shirt = Image.open(selected_shirt).convert("RGBA")

    bbox = get_shirt_bbox(shirt)

    if bbox:
        sx, sy, sw, sh = bbox
        shirt_crop = shirt.crop((sx, sy, sx + sw, sy + sh))

        scale = min(sw / design.width, sh / design.height) * padding_ratio
        new_size = (int(design.width * scale), int(design.height * scale))
        design = design.resize(new_size)

        disp_map = generate_displacement_map(shirt_crop).resize(new_size)
        design = apply_displacement(design, disp_map, displacement_strength)
        design = apply_shadows(design, shirt_crop)

        x = sx + (sw - design.width) // 2 + int(sw * horizontal_offset / 100)
        y = sy + int(sh * vertical_offset / 100)

        preview = shirt.copy()
        preview.paste(design, (x, y), design)

        st.image(preview, use_container_width=True)

        with st.expander("🧠 Auto-Generated Displacement Map"):
            st.image(disp_map, caption="Fabric fold map")

# ---------------- GENERATE ALL ----------------
if st.button("🚀 Generate Mockups (All Designs × All Shirts)"):
    if not (design_files and shirt_files):
        st.warning("Upload designs and shirts first.")
    else:
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for d_file in design_files:
                design_orig = Image.open(d_file).convert("RGBA")

                for s_file in shirt_files:
                    shirt = Image.open(s_file).convert("RGBA")
                    bbox = get_shirt_bbox(shirt)
                    if not bbox:
                        continue

                    sx, sy, sw, sh = bbox
                    shirt_crop = shirt.crop((sx, sy, sx + sw, sy + sh))

                    scale = min(sw / design_orig.width, sh / design_orig.height) * padding_ratio
                    new_size = (int(design_orig.width * scale), int(design_orig.height * scale))
                    design = design_orig.resize(new_size)

                    disp_map = generate_displacement_map(shirt_crop).resize(new_size)
                    design = apply_displacement(design, disp_map, displacement_strength)
                    design = apply_shadows(design, shirt_crop)

                    x = sx + (sw - design.width) // 2
                    y = sy + int(sh * vertical_offset / 100)

                    final_img = shirt.copy()
                    final_img.paste(design, (x, y), design)

                    img_bytes = io.BytesIO()
                    final_img.save(img_bytes, format="PNG")

                    zipf.writestr(
                        f"{os.path.splitext(d_file.name)[0]}_{os.path.splitext(s_file.name)[0]}.png",
                        img_bytes.getvalue()
                    )

        zip_buffer.seek(0)
        st.download_button(
            "📦 Download ZIP",
            zip_buffer,
            "realistic_mockups.zip",
            "application/zip"
        )
