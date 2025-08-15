import base64
import random
import cv2
import numpy as np
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from skimage.metrics import structural_similarity as ssim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SAFE_MESSAGES = [
    "Your snack is safe... for now. 🕵️‍♂️",
    "Looks untouched. You must have a very trustworthy environment. Or just boring snacks.",
    "All clear. The snack integrity field is stable.",
]
BREACH_MESSAGES = [
    "BREACH DETECTED! Someone's been nibbling! 🚨",
    "Houston, we have a problem. And it's missing a cookie.",
    "Evidence of tampering found. Unleash the hounds!",
]

# --- FINAL CORE IMAGE COMPARISON LOGIC ---
# This version creates a "ghost" of only the missing part.
def compare_images(image_a, image_b):
    # Convert images to grayscale for comparison.
    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    # Calculate the Structural Similarity Index (SSIM) to get a score and difference map.
    (score, diff) = ssim(gray_a, gray_b, full=True)

    # Decide if there's a breach based on the similarity score.
    if score < 0.9: # You can adjust this threshold (0.0 to 1.0)
        status = "BREACH"
        message = random.choice(BREACH_MESSAGES)
    else:
        status = "SAFE"
        message = random.choice(SAFE_MESSAGES)

    # Start with the test image as our base.
    output_image = image_b.copy()

    # If a breach is detected, we create and overlay the ghost.
    if status == "BREACH":
        # 1. Create a black and white mask of the differences.
        diff_map = (diff * 255).astype("uint8")
        thresh = cv2.threshold(diff_map, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        # 2. Clean the mask to remove noise and keep only significant changes.
        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # --- GHOST CREATION ---
        # 3. Use the cleaned mask to "cut out" the corresponding part from the ORIGINAL image.
        # This creates an image that is black everywhere except for the missing snack part.
        ghost_image = cv2.bitwise_and(image_a, image_a, mask=cleaned_mask)
        
        # 4. Blend this "ghost" onto our output image.
        # The ghost is made 40% opaque (0.4). The output_image starts at 100% opaque (1.0).
        # Because ghost_image is black in non-missing areas, it doesn't affect them.
        output_image = cv2.addWeighted(output_image, 1.0, ghost_image, 0.4, 0)

    return status, message, score, output_image

# --- Helper function to read images (no changes here) ---
def read_image_from_base64(base64_string: str):
    encoded_data = base64_string.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

# --- API Endpoint (no changes here) ---
@app.post("/analyze/")
async def analyze_snack(baseline_image_b64: str = Form(...), test_image_b64: str = Form(...)):
    baseline_img = read_image_from_base64(baseline_image_b64)
    test_img = read_image_from_base64(test_image_b64)
    
    h, w, _ = baseline_img.shape
    test_img = cv2.resize(test_img, (w, h))

    status, message, score, diff_image = compare_images(baseline_img, test_img)

    _, buffer = cv2.imencode('.png', diff_image)
    diff_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": status,
        "message": message,
        "ssim_score": round(score, 4),
        "diff_image_b64": f"data:image/png;base64,{diff_image_b64}",
    }