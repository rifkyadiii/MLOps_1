# Gunakan image resmi TensorFlow Serving
FROM tensorflow/serving:latest

# Set variabel lingkungan untuk nama model
ENV MODEL_NAME=mobile-price-model

# Salin model yang sudah di-train dari folder lokal ke dalam container
COPY ./serving_model/ /models/${MODEL_NAME}