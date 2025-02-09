using Images, Flux, FileIO

# Function to load and preprocess the image
function load_image(path::String)
    img = load(path)                   # Load the image
    img = Gray.(img)                    # Convert to grayscale
    img = imresize(img, (28, 28))       # Resize to 28x28
    img_tensor = permutedims(channelview(img), (2, 3, 1))  # Convert to tensor format
    return Float32.(img_tensor)         # Convert to Float32 type
end
