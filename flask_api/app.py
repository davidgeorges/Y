from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
import ssl
import collections

app = Flask(__name__)
CORS(app)

# Charger le modèle une seule fois
model = tf.keras.models.load_model('model/model.h5')

# Mapping des émotions
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
    img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # Ajoute la dimension canal
    img = np.expand_dims(img, axis=0)   # Ajoute la dimension batch
    return img

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer toutes les images peu importe leur clé
    files = [file for key, file in request.files.items()]
    
    if not files:
        return jsonify({"error": "No valid images received"}), 400

    emotion_count = collections.Counter()  # Stocker les émotions détectées

    try:
        for file in files:
            img_bytes = file.read()
            image = prepare_image(img_bytes)
            preds = model.predict(image)
            emotion_idx = np.argmax(preds[0])
            emotion = emotion_dict.get(emotion_idx, "Unknown")
            emotion_count[emotion] += 1

        # Trouver l'émotion la plus présente
        most_common_emotion = emotion_count.most_common(1)[0][0]

        return jsonify({
            "most_common_emotion": most_common_emotion,
            "emotion_counts": dict(emotion_count)
        })
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