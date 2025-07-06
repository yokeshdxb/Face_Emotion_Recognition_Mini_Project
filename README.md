# 😄 Face Emotion Recognition using Deep Learning (FER2013 Dataset)

This project focuses on building a Convolutional Neural Network (CNN) model that detects facial emotions using grayscale images from the FER2013 dataset. The model classifies images into 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## 📂 Dataset

- **Source**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- The dataset contains grayscale images of size `48x48` pixels categorized into 7 emotion classes.
- It includes:
  - Training set
  - Validation set (created from training using validation split)
  - Test set

---

## 🛠️ Project Workflow

### 📌 1. Install Dependencies
bash
!pip install kaggle tensorflow matplotlib seaborn --quiet
📌 2. Download Dataset
Upload your kaggle.json API key.

Automatically fetch and extract FER2013 dataset from Kaggle.

📌 3. Image Preprocessing
Used ImageDataGenerator for:

Rescaling

Data augmentation (rotation, zoom, shift, shear, flip)

Training/validation split

📌 4. Model Architecture
A custom CNN model using:

Conv2D, MaxPooling2D, Dropout, BatchNormalization, and Dense layers

CategoricalCrossentropy loss with label_smoothing

Adam optimizer

EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint for training optimization.

📌 5. Evaluation
Accuracy and loss curves plotted

Final evaluation on test dataset

📌 6. Inference
Custom function to predict emotion from a single image

Displays both actual and predicted labels for better explainability

🧠Model Summary
Conv2D → BatchNorm → MaxPooling → Dropout  
Conv2D → BatchNorm → MaxPooling → Dropout  
Conv2D → BatchNorm → MaxPooling → Dropout  
Flatten → Dense(512) → Dropout → Output(7 classes)

📊 Results
Final model was trained for up to 300 epochs

Achieved competitive accuracy on both validation and test sets

Inference tested on images from test set showing accurate predictions

🔍 Sample Inference Output
predict_emotion("/content/test/happy/PrivateTest_13103594.jpg")
Actual Label	Predicted Label
Happy	Happy ✅

💻 Tech Stack
Python

TensorFlow / Keras

Google Colab

Matplotlib & Seaborn

Kaggle API

ImageDataGenerator

🙌 Mentions
Special thanks to my mentors: Raja, Pranav, Chirag, Sandhya

📁 License
This project is for educational purposes only. Dataset is publicly available on Kaggle under its own terms.

⭐️ If you found this helpful...
Please give a ⭐️ to the repository and feel free to fork it!
