#Y - IA - Lea (M1), David (M2), Frederic (M2)


# Facial Expression Recognition API

Il s'agit d'une API basée sur Flask pour la reconnaissance des expressions faciales. Elle reçoit un fichier image, le traite à l'aide d'un modèle CNN pré-entraîné et renvoie l'émotion prédite avec son niveau de confiance. Le modèle s'attend à des images en niveaux de gris redimensionnées à 48×48 pixels.

---

## Table des matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Endpoints de l'API](#endpoints-de-lapi)
- [Test](#testing)

---

## Installation

### Prerequisites

- Python 3.9 -- 3.11
- [pip](https://pip.pypa.io/en/stable/)

### Required Libraries

Install the necessary libraries using pip:

```bash
pip install requirements.txt
```

## Utilisation

```bash
python app.py
```

## Endpoints de l'API

Cette API, développée avec Flask, permet de reconnaître des expressions faciales à partir d'une image. Le modèle pré-entraîné attend une image en niveaux de gris de 48×48 pixels et renvoie une prédiction parmi sept émotions : **Anger, Disgust, Fear, Happy, Sad, Surprise,** et **Neutral**. La réponse inclut également la confiance (probabilité) associée à la prédiction.

---

### POST /predict

**Description :**  
Cet endpoint reçoit une image, la convertit en niveaux de gris et la redimensionne à 48×48 pixels, puis renvoie l'émotion prédite ainsi que la confiance (probabilité) associée.

**URL :**  
`http://127.0.0.1:5000/predict`

**Méthode :**  
`POST`

**Paramètres :**

- **image (file, requis) :**  
  Le fichier image à analyser. Le format doit être un format d'image standard (JPEG, PNG, etc.). L'image est automatiquement convertie en niveaux de gris et redimensionnée.

**Réponses :**

- **Succès (HTTP 200) :**

```json
{
  "emotion_counts": {"Anger": 3},
  "most_common_emotion": "Anger"
}
```

## Testing

Pour tester l'API avec Postman, suivez ces étapes :

**Ouvrir Postman et créer une nouvelle requête :**
- Lancez Postman et cliquez sur "New" puis sélectionnez "Request".

**Configurer la requête :**
- Méthode : Choisissez POST.
- URL : Entrez http://127.0.0.1:5000/predict

**Configurer le corps de la requête :**
- Cliquez sur l'onglet Body.
- Sélectionnez form-data.
- Ajoutez une clé nommée image.
- Dans la colonne "Type" (située à droite de la clé), choisissez File.
- Cliquez sur la case "Value" pour sélectionner un fichier image sur votre ordinateur (formats acceptés : JPEG, PNG, etc.).

**Envoyer la requête :**
- Cliquez sur le bouton Send pour envoyer la requête à votre API.

**Vérifier la réponse :**
- Postman affichera la réponse JSON. Vous devriez voir un résultat similaire à :

    ```json
    {
      "emotion_counts": {"Anger": 3},
      "most_common_emotion": "Anger"
    }
    ```
## Deployment

