import os
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Paths
benefits_csv_path = r"D:\\TENSOR EXPO\\benefits.csv"
model_path = r"D:\\TENSOR EXPO\\medicinal_leaf_model.h5"

# Load benefits from CSV into a dictionary
def load_benefits(csv_path):
    benefits = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                leaf_name, benefit = row[0].strip(), row[1].strip()  # Strip any extra spaces
                benefits[leaf_name] = benefit
    return benefits

benefits_dict = load_benefits(benefits_csv_path)

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

model.summary()

# Prediction function
def predict_leaf(image_path, model, benefits_dict):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))  
    img_array = img_to_array(img) / 255.0 
    img_array = tf.expand_dims(img_array, axis=0) 

    # Make prediction
    predictions = model.predict(img_array)
    class_index = tf.argmax(predictions[0]).numpy()

    # Get predicted leaf name
    predicted_label = class_indices.get(class_index, "Unknown Leaf")

    # Check if predicted leaf name exists in the benefits dictionary
    if predicted_label in benefits_dict:
        benefits = benefits_dict[predicted_label]
    else:
        benefits = "Benefits not found."

    return predicted_label, benefits, predictions[0]

# Example Testing
image_path = r"D:\TENSOR EXPO\combined_dataset\leaf_Betel\1880.jpg"  # Path to your test image
predicted_label, benefits, prediction_probs = predict_leaf(image_path, model, benefits_dict)

# Displaying the image
img = load_img(image_path)
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.title(f"Predicted: {predicted_label}")
plt.show()

#prediction
print(f"Predicted Leaf: {predicted_label}")
print(f"Medicinal Benefits: {benefits}")

# Plotting the prediction probabilities
plt.figure(figsize=(8, 6))
plt.bar(range(len(prediction_probs)), prediction_probs)
plt.title("Prediction Probabilities")
plt.xlabel("Class Index")
plt.ylabel("Probability")
plt.show()
