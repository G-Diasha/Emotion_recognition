# Emotion Recognition using ResNet-50 (Transfer Learning)

**🌐Live APP: _https://emotionrecognition-dg.streamlit.app/_**

🚀**Project Overview**

This project implements a Deep Learning–based Emotion Recognition system capable of identifying human emotions from facial images. 
The goal of the project was to explore transfer learning with ResNet-50 and build a complete end-to-end machine learning pipeline, 
from model development to deployment. The model was trained using the AffectNet dataset, with additional techniques applied to address 
class imbalance and overfitting. The model is deployed as an interactive web application using Streamlit and containerized with Docker for 
portability and reproducibility.

🧠 **Tech Stack**

1) Language - Python
2) TensorFlow / Keras – Deep learning framework
3) ResNet-50 – Pretrained convolutional neural network
4) Scikit-learn- Model evaluation utilites
5) NumPy
6) Streamlit – Web application interface
7) Docker – Containerization for reproducible deployment

📊**Dataset**
The dataset includes images labeled with emotions such as: Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral

**Challenge: Dataset Imbalance**
A key challenge in the dataset was class imbalance, where some emotions appeared significantly more frequently than others.
To mitigate this issue, the following techniques were applied: **Class weighting** & **Data augmentation**
These methods helped the model learn more balanced representations across emotion classes.

**Model Architecture**

The project uses ResNet-50 with transfer learning. ResNet-50 introduces residual connections, which help very deep neural networks 
to train effectively by mitigating the vanishing gradient problem. Instead of training the model from scratch, a pretrained ResNet-50 model 
was used as a feature extractor, and the top layers were fine-tuned for emotion classification.

🧪**Model Pipeline**

The complete machine learning pipeline consists of the following steps:

**1️⃣ Data Preprocessing**

a) Image resizing

b) Normalization

c) Train / validation split

**2️⃣ Handling Class Imbalance & Improving Generalization**
Augmentation techniques included:

a) Random flips

b)Rotation

c)Zoom transformations

**Class weighting** calculated for each class using scikit-learn "class_compute" function and added during model training to ensure 
underrepresented emotions were learned effectively. 

**3️⃣ Transfer Learning**

a) Load pretrained ResNet-50

b) Freeze early convolutional layers

c) Add custom classification head

**4️⃣ Model Training**

The model was trained using TensorFlow/Keras, optimizing for classification accuracy while monitoring validation performance.

**5️⃣ Model Evaluation**
Model performance was evaluated using training and validation accuracy metrics.
Initial results indicated strong overfitting:

**Training Accuracy: 95%**

**Validation Accuracy: 58%**

After applying class weighting and data augmentation, the model achieved improved generalization:

**Training Accuracy: 78%**

**Validation Accuracy: 68%**

This improvement indicated better generalization to unseen data. The model performs well on clearly distinguishable facial expressions, 
although some confusion remains between visually similar emotions such as disgust vs angry and fear vs sad.
However, the model reliably distinguishes clearly different expressions (e.g., a smiling face is not classified as sad or angry), 
suggesting that it has learned meaningful emotional features.

**Deployment**

The trained model was deployed as an interactive web application.

**🌐 Streamlit Application**
The Streamlit interface allows users to upload a facial image -> run emotion prediction -> view the predicted emotion

**🧩 Docker Containerization**
The application was also containerized using Docker to ensure:
1) Reproducible environments

2) Simplified deployment

3) Portability across systems

**🛠️ Running the Project Locally**

***1.Clone the Repository***

_git clone https://github.com/G-Diasha/emotion-recognition.git_

***2.Install dependencies***

_pip install -r requirements.txt_

***3.Run the streamlit app***

_streamlit run app.py_

**Docker Deployment**

***1.Build the Docker image:***

_docker build -t emotion-recognition ._

***2.Run the container***

_docker run -p 8501:8501 emotion-recognition_
