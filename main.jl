using Flux
using Images, FileIO  # Ensure dependencies are loaded

# Include model and preprocessing scripts
include("model.jl")
include("preprocess.jl")

# Load the model
model = build_model()

# Load and preprocess the image
img_tensor = load_image("image1.jpg")

# Ensure the image tensor has the correct dimensions (Height, Width, Channels, Batch)
img_tensor = reshape(img_tensor, (28, 28, 1, 1))  # Adjust this shape according to your model

# Forward pass through the model
output = model(img_tensor)

# Print the model output
println("Model output: ", output)
