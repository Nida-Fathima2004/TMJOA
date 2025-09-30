import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import tempfile
import io
from ultralytics import YOLO

model = YOLO("best.pt")
model.model = torch.load("best.pt", map_location="cpu", weights_only=False)

# ---------------------------
# Streamlit page setup
# ---------------------------
st.set_page_config(page_title="TMJ Symmetry Checker", layout="wide")
st.title("🦷 TMJ Symmetry Checker")
st.markdown("Upload an image to check for symmetry/deformation of TMJ.")


# ---------------------------
# Function to analyze image
# ---------------------------
def analyze_image(image_path, threshold=5):
    image = cv2.imread(image_path)
    results = model.predict(source=image, conf=0.25, verbose=False)
    metrics = {"status": "Not processed"}

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) < 2:
            metrics["status"] = f"Only {len(boxes)} objects detected — need 2."
            return image, metrics

        # take only 2 boxes (left & right)
        boxes = sorted(boxes, key=lambda b: b[0])[:2]
        left_box, right_box = boxes

        # ---------------------------
        # Symmetry calculations (updated strategy)
        # ---------------------------
        left_cx = (left_box[0] + left_box[2]) / 2
        right_cx = (right_box[0] + right_box[2]) / 2
        left_w = left_box[2] - left_box[0]
        right_w = right_box[2] - right_box[0]
        left_h = left_box[3] - left_box[1]
        right_h = right_box[3] - right_box[1]

        image_center_x = image.shape[1] / 2
        left_distance = abs(left_cx - image_center_x)
        right_distance = abs(right_cx - image_center_x)

        symmetry_error = abs(left_distance - right_distance)
        max_possible_distance = image_center_x
        asymmetry_percent = (symmetry_error / max_possible_distance) * 100   # <-- Asymmetry %

        width_diff = abs(left_w - right_w) / max(left_w, right_w) * 100      # <-- Width %
        height_diff = abs(left_h - right_h) / max(left_h, right_h) * 100     # <-- Height %

        # build metrics
        metrics = {
            "asymmetry": round(asymmetry_percent, 2),
            "width_diff": round(width_diff, 2),
            "height_diff": round(height_diff, 2),
            "status": "⚠️ Deformation Found"
            if asymmetry_percent > threshold
            else "✅ No Deformation",
        }

        # draw boxes + labels
        for box, label in zip([left_box, right_box], ["Left", "Right"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # overlay metrics
        y_offset = 30
        overlay_text = [
            f"Asymmetry: {metrics['asymmetry']}%",
            f"Width Diff: {metrics['width_diff']}%",
            f"Height Diff: {metrics['height_diff']}%",
            f"Status: {metrics['status']}",
        ]
        for line in overlay_text:
            cv2.putText(
                image,
                line,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            y_offset += 30

    return image, metrics


# ---------------------------
# File uploader & processing
# ---------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("Analyzing image..."):
        processed_image, metrics = analyze_image(tmp_path, threshold=5)

    # convert images for display
    original_image = Image.open(tmp_path)
    processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # show images
    st.subheader("🔹 Images")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_column_width=True)
    with col2:
        st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)

    # show metrics
    st.subheader("📊 Analysis Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", metrics.get("status", "N/A"))
    col2.metric("Asymmetry (%)", metrics.get("asymmetry", 0))
    col3.metric("Width Diff (%)", metrics.get("width_diff", 0))
    col4.metric("Height Diff (%)", metrics.get("height_diff", 0))

    # download button
    processed_pil = Image.fromarray(processed_image_rgb)
    buffer = io.BytesIO()
    processed_pil.save(buffer, format="PNG")
    buffer.seek(0)

    st.download_button(
        label="💾 Download Processed Image",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png",
    )

    # final message
    if "⚠️" in metrics["status"]:
        st.warning("Deformation detected! Please consult a specialist.")
    else:
        st.success("No significant deformation detected.")
