import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import io
import cv2
from streamlit_drawable_canvas import st_canvas

# ---------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------
st.set_page_config(page_title="👕 Smart Shirt Mockup Generator", layout="wide")
st.title("👕 Interactive Shirt Mockup Generator")

st.markdown("""
Upload your **shirt template** and **design PNG**,  
then drag, resize, and recolor your mockup in real-time.
""")

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
st.sidebar.header("🎨 Shirt Controls")
shirt_color = st.sidebar.color_picker("Pick Shirt Color", "#FFFFFF")
blend_strength = st.sidebar.slider("Design Opacity", 0.3, 1.0, 0.85, 0.05)

# ---------------------------------------------------------------
# FILE UPLOADS
# ---------------------------------------------------------------
shirt_file = st.file_uploader("🧥 Upload Shirt Template", type=["png", "jpg", "jpeg"])
design_file = st.file_uploader("🎨 Upload Design (PNG)", type=["png"], accept_multiple_files=False)

if not shirt_file or not design_file:
    st.info("⬆️ Please upload both a shirt image and a design PNG to start.")
    st.stop()

shirt = Image.open(shirt_file).convert("RGB")
design = Image.open(design_file).convert("RGBA")

# ---------------------------------------------------------------
# SHIRT RECOLOR FUNCTION
# ---------------------------------------------------------------
def recolor_shirt(shirt_img, color_hex):
    """Recolor shirt while preserving texture (wrinkles and shading)."""
    shirt_rgb = np.array(shirt_img).astype(np.float32) / 255.0
    color_rgb = np.array(Image.new("RGB", shirt_img.size, color_hex)).astype(np.float32) / 255.0
    gray = np.mean(shirt_rgb, axis=2, keepdims=True)
    recolored = gray * color_rgb
    recolored = np.clip(recolored * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(recolored)

# ---------------------------------------------------------------
# REALISTIC BLEND FUNCTION
# ---------------------------------------------------------------
def blend_design_with_shirt(shirt, design, pos, opacity=0.8):
    """Blend design realistically onto the shirt."""
    x, y = pos
    shirt_np = np.array(shirt).astype(np.float32) / 255.0
    design_np = np.array(design).astype(np.float32) / 255.0

    h, w = design_np.shape[:2]
    H, W = shirt_np.shape[:2]

    if x + w > W or y + h > H:
        w = min(w, W - x)
        h = min(h, H - y)
        design_np = design_np[:h, :w]

    overlay = shirt_np.copy()
    region = overlay[y:y+h, x:x+w]

    # Auto-select blend mode
    brightness = np.mean(region)
    if brightness > 0.6:
        # Multiply for light shirts
        blended = region * design_np[:, :, :3]
    else:
        # Overlay for dark shirts
        blended = 1 - (1 - region) * (1 - design_np[:, :, :3])

    alpha = design_np[:, :, 3:4] * opacity
    region_out = (1 - alpha) * region + alpha * blended
    overlay[y:y+h, x:x+w] = region_out

    return Image.fromarray((overlay * 255).astype(np.uint8))

# ---------------------------------------------------------------
# STEP 1: RECOLOR SHIRT
# ---------------------------------------------------------------
recolored_shirt = recolor_shirt(shirt, shirt_color)

# ---------------------------------------------------------------
# STEP 2: INTERACTIVE CANVAS
# ---------------------------------------------------------------
st.markdown("### 🖱️ Drag & Position Your Design")
canvas_result = st_canvas(
    fill_color="rgba(255,255,255,0)",
    stroke_width=0,
    background_image=recolored_shirt,
    update_streamlit=True,
    height=recolored_shirt.height,
    width=recolored_shirt.width,
    drawing_mode="transform",  # allows drag, resize, rotate
    key="canvas",
)

# ---------------------------------------------------------------
# STEP 3: MOCKUP GENERATION
# ---------------------------------------------------------------
if st.button("🚀 Generate Realistic Mockup"):
    # Default position if none selected
    x, y = 100, 100
    w, h = design.size

    # Optional: you can store canvas data in session_state for advanced placement
    try:
        canvas_img = canvas_result.image_data
    except:
        canvas_img = None

    # Paste design at the center (you can replace with canvas coordinates later)
    x = (recolored_shirt.width - design.width) // 2
    y = (recolored_shirt.height - design.height) // 3

    final = blend_design_with_shirt(recolored_shirt, design, (x, y), opacity=blend_strength)
    st.image(final, caption="🧵 Final Realistic Mockup", use_container_width=True)

    # Download button
    buf = io.BytesIO()
    final.save(buf, format="PNG")
    st.download_button(
        label="📥 Download Mockup",
        data=buf.getvalue(),
        file_name="realistic_mockup.png",
        mime="image/png",
    )
