import streamlit as st
from PIL import Image, ImageChops
import numpy as np
import cv2
import zipfile
import io
import os

st.set_page_config(page_title="Realistic Shirt Mockup Generator", layout="centered")
st.title("👕 Realistic Shirt Mockup Generator (Fabric Folds Enabled)")

# ---------------- SIDEBAR ----------------
padding_ratio = st.sidebar.slider("Design Size", 0.1, 1.0, 0.45, 0.05)
vertical_offset = st.sidebar.slider("Vertical Offset (%)", -50, 100, 30, 1)
horizontal_offset = st.sidebar.slider("Horizontal Offset (%)", -50, 50, 0, 1)
displacement_strength = st.sidebar.slider("Fabric Fold Strength", 0, 50, 20, 1)

# ---------------- UPLOAD ----------------
design_files = st.file_uploader("Upload Design PNGs", type=["png"], accept_multiple_files=True)
shirt_files = st.file_uploader("Upload Shirt Images", type=["png"], accept_multiple_files=True)

disp_file = st.file_uploader("Upload Displacement Map (Grayscale PNG)", type=["png"])
texture_file = st.file_uploader("Upload Fabric Texture (Optional)", type=["png"])

# ---------------- FUNCTIONS ----------------
def get_shirt_bbox(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.boundingRect(max(contours, key=cv2.contourArea))
    return None


def apply_displacement(design, disp_map, strength):
    design_np = np.array(design)
    disp = np.array(disp_map.convert("L"))

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


def blend_with_texture(design, texture):
    texture = texture.resize(design.size).convert("RGBA")
    return ImageChops.multiply(design, texture)


def apply_shadows(design, shirt):
    shadows = shirt.convert("L").resize(design.size)
    shadows = Image.merge("RGBA", (shadows, shadows, shadows, shadows))
    return ImageChops.multiply(design, shadows)

# ---------------- PREVIEW ----------------
if design_files and shirt_files and disp_file:
    design = Image.open(design_files[0]).convert("RGBA")
    shirt = Image.open(shirt_files[0]).convert("RGBA")
    disp_map = Image.open(disp_file)

    bbox = get_shirt_bbox(shirt)
    if bbox:
        sx, sy, sw, sh = bbox
        scale = min(sw / design.width, sh / design.height) * padding_ratio
        new_size = (int(design.width * scale), int(design.height * scale))
        design = design.resize(new_size)

        design = apply_displacement(design, disp_map.resize(new_size), displacement_strength)

        if texture_file:
            texture = Image.open(texture_file)
            design = blend_with_texture(design, texture)

        design = apply_shadows(design, shirt)

        x = sx + (sw - design.width) // 2 + int(sw * horizontal_offset / 100)
        y = sy + int(sh * vertical_offset / 100)

        preview = shirt.copy()
        preview.paste(design, (x, y), design)

        st.image(preview, caption="Live Realistic Preview", use_container_width=True)

# ---------------- GENERATE ----------------
if st.button("🚀 Generate Realistic Mockups"):
    if not (design_files and shirt_files and disp_file):
        st.warning("Upload designs, shirts, and a displacement map.")
    else:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zipf:
            disp_map = Image.open(disp_file)

            for design_file in design_files:
                design = Image.open(design_file).convert("RGBA")

                for shirt_file in shirt_files:
                    shirt = Image.open(shirt_file).convert("RGBA")
                    bbox = get_shirt_bbox(shirt)
                    if not bbox:
                        continue

                    sx, sy, sw, sh = bbox
                    scale = min(sw / design.width, sh / design.height) * padding_ratio
                    new_size = (int(design.width * scale), int(design.height * scale))
                    d = design.resize(new_size)

                    d = apply_displacement(d, disp_map.resize(new_size), displacement_strength)

                    if texture_file:
                        texture = Image.open(texture_file)
                        d = blend_with_texture(d, texture)

                    d = apply_shadows(d, shirt)

                    x = sx + (sw - d.width) // 2
                    y = sy + int(sh * vertical_offset / 100)

                    result = shirt.copy()
                    result.paste(d, (x, y), d)

                    img_bytes = io.BytesIO()
                    result.save(img_bytes, format="PNG")
                    zipf.writestr(
                        f"{os.path.splitext(design_file.name)[0]}_{shirt_file.name}",
                        img_bytes.getvalue()
                    )

        zip_buffer.seek(0)
        st.download_button(
            "📦 Download Mockups",
            zip_buffer,
            "realistic_mockups.zip",
            "application/zip"
        )
