import torch
import time
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os

# Paths to your models and the test images
model_paths = ["C:/junha/Personal_Notebook/models/efficientNet_OverSampling.pth",
               "C:/junha/Personal_Notebook/models/efficientNetV2_OverSampling.pth",
               "C:/junha/Personal_Notebook/models/swinTransformer_OverSampling.pth",
               "C:/junha/Personal_Notebook/models/swinTransformer_Tiny_OverSampling.pth"]

image_folder = "C:/junha/Personal_Notebook/oversampling_data/z_keep_out_for_final_test/normal"

# Preprocess for the images
transform = transforms.Compose([
    transforms.Resize((2352, 832)),  # Ensure size matches the model's input requirements
    transforms.ToTensor()
])


# Function to classify an image with a specific model and measure time
def classify_image(image_path, model):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    start_time = time.time()
    with torch.no_grad():
        output = model(image)
    end_time = time.time()
    classification_time = end_time - start_time
    return classification_time


# Plot classification times for each model
plt.figure(figsize=(10, 6))

for model_path in model_paths:
    # Load the model
    model = torch.load(model_path)
    model.eval()

    # Classify each image and record the time
    classification_times = []
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        time_taken = classify_image(image_path, model)
        classification_times.append(time_taken)

    # Plot the classification times for this model
    model_name = os.path.basename(model_path)
    plt.plot(classification_times, marker='o', linestyle='-', label=model_name)

# Configure the plot
plt.xlabel("Image Index")
plt.ylabel("Time Taken (seconds)")
plt.title("Classification Time per Image for Different Models")
plt.legend()
plt.show()
