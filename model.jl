using Flux

# Define a simple CNN model
function build_model()
    return Chain(
        Conv((3, 3), 1 => 8, relu),  # Convolution layer with ReLU
        MaxPool((2,2)),              # Downsampling
        Conv((3, 3), 8 => 16, relu), # Second convolution
        MaxPool((2,2)),              # Another downsampling
        Flux.flatten,                # Flatten the output
        Dense(64, 10),               # Fully connected layer
        softmax                      # Output activation
    )
end
