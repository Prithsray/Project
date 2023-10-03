from flask import Flask, request, render_template, redirect, url_for,jsonify
from tensorflow import keras
from PIL import Image
import numpy as np
import io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mysql.connector
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('D:\COLLEGEMATERIALS\Project Me\spine_fracture_detection\VGG16_MRI_classification.h5')

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]

# Define a route for image classification

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Image_Classification')
def Image():
    return render_template("Image_Classification.html")


@app.route('/Profile_Recomedation')
def Profile():
    return render_template("profile_recommender.html")


@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        image = request.files['image']

        # Receive an image file from the client
        if image:
            # Read the content of the uploaded file
            image_content = image.read()
            
            # Use PIL to open and resize the image
            image_pil = Image.open(io.BytesIO(image_content))
            new_width = 64  # Set the desired width
            new_height = 64  # Set the desired height
            image_pil = image_pil.resize((new_width, new_height))
            
            # Convert the PIL Image to a NumPy array
            image_np = np.array(image_pil)

            # Expand dimensions to match the expected shape for the model
            image_np = np.expand_dims(image_np, axis=0)

            # Preprocess the image using ImageDataGenerator
            image_preprocessed = datagen.flow(image_np, shuffle=False).next()

            # Perform inference with the preprocessed image
            result = model.predict(image_preprocessed)

            # Process the result as needed
            return jsonify({'result': result.tolist()})
        else:
            return jsonify({'error': 'No file provided'})

    except Exception as e:
        return jsonify({'error': str(e)})


# Database configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="recommendation"
)

cursor = db.cursor()

# Create a table to store job profiles if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS job_profiles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        profile_name VARCHAR(255),
        lawyer_type VARCHAR(255),            
        description TEXT
    )
""")
db.commit()

# Load the job profiles dataset from the database
cursor.execute("SELECT * FROM job_profiles")
job_profiles_data = cursor.fetchall()
job_profiles = pd.DataFrame(job_profiles_data, columns=['id', 'title', 'lawyer_type', 'description'])

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

# Function to get BERT embeddings for text
def get_bert_embeddings(text):
    input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# Function to get profile recommendations based on BERT embeddings
def get_profile_recommendations(input_description):
    input_description = input_description.lower()
    if job_profiles.empty:
        return []  # Return an empty list if there are no profiles in the database
    input_embedding = get_bert_embeddings(input_description)
    job_embeddings = [get_bert_embeddings(desc) for desc in job_profiles['lawyer_type']]
    
    # Calculate cosine similarity between input and job profiles
    similarities = [torch.cosine_similarity(input_embedding, job_embedding) for job_embedding in job_embeddings]
    
    # Sort by similarity scores in descending order
    profile_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    
    return job_profiles['title'].iloc[profile_indices[:5]]


@app.route('/upload', methods=['POST'])
def upload_profile():
    if request.method == 'POST':
        profile_name = request.form['profile_name']
        lawyer_type = request.form['lawyer_type']
        description = request.form['description']
        cursor.execute("INSERT INTO job_profiles (profile_name,lawyer_type, description) VALUES (%s,%s, %s)", (profile_name, lawyer_type, description))
        db.commit()
    return redirect(url_for('index'))

@app.route('/recommend', methods=['POST'])
def recommend_profiles():
    if request.method == 'POST':
        profile_description = request.form['profile_description']
        recommended_profiles = get_profile_recommendations(profile_description)
        return render_template('recommendations.html', profile_description=profile_description, recommended_profiles=recommended_profiles)


if __name__ == '__main__':
    app.run(debug=True)
