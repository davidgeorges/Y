from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import ssl

app = Flask(__name__)
CORS(app)

# Load the model once when the app starts
model = tf.keras.models.load_model('model/model.h5')

# Dictionary mapping indices to emotions
emotion_dict = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

def prepare_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    # Read as grayscale (1 channel) instead of color (3 channels)
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    # Resize to match the model’s expected input size
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    # Expand dimensions so that it becomes (1, 48, 48, 1)
    img = np.expand_dims(img, axis=-1)  # adds the channel dimension
    img = np.expand_dims(img, axis=0)   # adds the batch dimension
    return img


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400

    try:
        img_bytes = file.read()
        image = prepare_image(img_bytes)
        # Make a prediction with the model
        preds = model.predict(image)
        print(preds)
        emotion_idx = np.argmax(preds[0])
        emotion = emotion_dict.get(emotion_idx, "Unknown")
        confidence = float(preds[0][emotion_idx])

        return jsonify({"confidence": confidence,"emotion": emotion})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create an SSL context for mTLS
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.verify_mode = ssl.CERT_REQUIRED
    # Load the server certificate and key
    context.load_cert_chain(
        certfile="certs/server.crt",
        keyfile="certs/server.key"
    )
    # Load the CA certificate to verify client certificates
    context.load_verify_locations(cafile="certs/ca.crt")
    # Start the Flask server with mTLS
    app.run(host="0.0.0.0", port=5000, ssl_context=context, debug=True)