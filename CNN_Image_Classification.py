# Import required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from tensorflow.keras.layers import Input

# Initialize CNN
MesinKlasifikasi = Sequential([
    Input(shape=(128, 128, 3)),  # Explicit input layer
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(units=128, activation='relu'),
    Dense(units=1, activation='sigmoid')
])

# Compile CNN
MesinKlasifikasi.compile(optimizer='adam', 
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])

# Image Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and prepare training data
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                               target_size=(128, 128),
                                               batch_size=32,
                                               class_mode='binary')

# Load and prepare test data
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                          target_size=(128, 128),
                                          batch_size=32,
                                          class_mode='binary')

# Train the model - using fit() instead of fit_generator()
MesinKlasifikasi.fit(
    training_set,
    steps_per_epoch=8000//32,  # Using integer division
    epochs=50,
    validation_data=test_set,
    validation_steps=2000//32   # Using integer division
)

# Test the model on new images
def test_model_performance():
    count_dog = 0
    count_cat = 0
    
    # Test on 1000 dog images
    for i in range(4001, 5001):
        try:
            test_image = image.load_img(f'dataset/test_set/dogs/dog.{i}.jpg', 
                                      target_size=(128, 128))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = MesinKlasifikasi.predict(test_image)
            
            if result[0][0] == 0:
                prediction = 'cat'
                count_cat += 1
            else:
                prediction = 'dog'
                count_dog += 1
        except Exception as e:
            print(f"Error processing image {i}: {str(e)}")
            continue
    
    print("Results for 1000 dog images:")
    print(f"Predicted as dog: {count_dog}")
    print(f"Predicted as cat: {count_cat}")
    print(f"Accuracy: {count_dog/10}%")

# Run the test
test_model_performance()