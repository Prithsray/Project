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
import tensorflow as tf
import numpy as np
import pickle
from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, render_template, request
from flask_mysqldb import MySQL
from flask import Flask, render_template, request, redirect, url_for, Response
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle,Image
from flask import Flask, render_template, request, redirect, url_for, Response
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, PageTemplate, Frame, Paragraph
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet


app = Flask(__name__)


#pages route

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/Image_Classification')
def Image_classify():
    return render_template("Image_Classification.html")


@app.route('/Profile_Recomedation')
def Profile():
    return render_template("profile_recommender.html")

@app.route('/Sentiment')
def Sentiment():
    return render_template("sentiment_analyse.html")







# FOR IMAGE CLASSIFICATION
# Load the trained model
model1 = keras.models.load_model('D:\COLLEGEMATERIALS\Project Me\Project\VGG16_MRI_classification.h5')

# Create an ImageDataGenerator for preprocessing
datagen = ImageDataGenerator(rescale=1./255)  # Normalize pixel values to [0, 1]

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
            result = model1.predict(image_preprocessed)

            # Process the result as needed
            return jsonify({'result': result.tolist()})
        else:
            return jsonify({'error': 'No file provided'})

    except Exception as e:
        return jsonify({'error': str(e)})








#FOR RECOMMENDATION SYSTEM

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









#TEXT SENTIMENT ANALYSER
model2 = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    if request.method == 'POST':
        text = request.form['text']

        # Tokenize and preprocess the input text using BERT tokenizer
        inputs = tokenizer2(text, padding=True, truncation=True, return_tensors="pt", max_length=100)
        
        # Perform sentiment analysis with BERT using the model directly
        outputs = model2(**inputs)

        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        label = torch.argmax(probabilities, dim=1).item()

        # Map the label to sentiment
       # labels = ['Negative', 'Positive']
        sentiment = "Positive" if label >0.5 else "Negative"


        return render_template('sentiment_analyse.html', sentiment=sentiment, input_text=text)









#CRUD

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'crud'
mysql = MySQL(app)


@app.route('/crud')
def index_00():
    # Retrieve records from the database
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM your_table')
    data = cur.fetchall()
    cur.close()
    return render_template('Basic_crud.html', records=data)

@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        cur = mysql.connection.cursor()
        cur.execute('INSERT INTO your_table (name, email) VALUES (%s, %s)', (name, email))
        mysql.connection.commit()
        cur.close()
        return redirect('/crud')

@app.route('/edit/<int:id>', methods=['GET', 'POST'])
def edit(id):
    cur = mysql.connection.cursor()
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        cur.execute('UPDATE your_table SET name = %s, email = %s WHERE id = %s', (name, email, id))
        mysql.connection.commit()
        cur.close()
        return redirect('/crud')
    else:
        cur.execute('SELECT * FROM your_table WHERE id = %s', (id,))
        data = cur.fetchone()
        cur.close()
        return render_template('edit.html', record=data)

@app.route('/delete/<int:id>', methods=['GET', 'POST'])
def delete(id):
    cur = mysql.connection.cursor()
    cur.execute('DELETE FROM your_table WHERE id = %s', (id,))
    mysql.connection.commit()
    cur.close()
    return redirect('/crud')

@app.route('/download_excel')
def download_excel():
    # Retrieve records from the database
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM your_table')
    data = cur.fetchall()
    cur.close()

    # Create a pandas DataFrame from the data
    df = pd.DataFrame(data, columns=['ID', 'Name', 'Email'])

    # Create an in-memory Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)

    excel_buffer.seek(0)

    # Serve the Excel file as a download
    response = Response(excel_buffer.read(), content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers["Content-Disposition"] = "attachment; filename=records.xlsx"

    return response


@app.route('/print_pdf')
def print_pdf():
    # Retrieve records from the database
    cur = mysql.connection.cursor()
    cur.execute('SELECT * FROM your_table')
    data = cur.fetchall()
    cur.close()

    # Create a list to store the data for the PDF table
    pdf_data = [['ID', 'Name', 'Email']]  # Header row
    pdf_data.extend(data)  # Data rows

    # Create a PDF file
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    elements = []

#create logo
   
    logo_path = os.path.join(app.root_path, 'static/logo.png')
    logo = Image(logo_path, width=100, height=100)
    elements.append(logo)

 # Add a heading line
    styles = getSampleStyleSheet()
    heading_text = "Records"  # Replace with your desired heading text
    heading = Paragraph(heading_text, styles['Title'])
    elements.append(heading)

    # Create a PDF table
    pdf_table = Table(pdf_data)
    pdf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),  # Header row background color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # left-align all cells
        ('TEXTCOLOR', (0, 0), (-1, 0), (1, 1, 1)),  # Header text color
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header font
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Header padding
        ('BACKGROUND', (0, 1), (-1, -1), (0.95, 0.95, 0.95)),  # Data row background color
        ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0)),  # Table grid
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),  # Data font
        ('FONT_SIZE', (0, 1), (-1, -1), 12),  # Data font size
    ]))

    elements.append(pdf_table)
    doc.build(elements)

    pdf_buffer.seek(0)

    # Serve the PDF file as a download
    response = Response(pdf_buffer.read(), content_type='application/pdf')
    response.headers["Content-Disposition"] = "inline; filename=records.pdf"

    return response


if __name__ == '__main__':
    app.run(debug=True)
