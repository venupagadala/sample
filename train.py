import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications import VGG16, VGG19
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
import os

warnings.filterwarnings('ignore')

# Define the path to your dataset
base_dir = 'COVID-19_Radiography_Dataset'
model_dir = 'models'

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Create the data generators
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),  # Reduced target_size to 128x128
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    subset='training'
)

validation_generator = validation_datagen.flow_from_directory(
    base_dir,
    target_size=(128, 128),  # Reduced target_size to 128x128
    batch_size=64,  # Increased batch size
    class_mode='categorical',
    subset='validation'
)

# Ensure the model directory exists
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Build the CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Load the VGG16 model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
x = base_model_vgg16.output
x = Flatten()(x)
x = Dense(4, activation='softmax')(x)
vgg16_model = Model(inputs=base_model_vgg16.input, outputs=x)

# Load the VGG19 model
base_model_vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
y = base_model_vgg19.output
y = Flatten()(y)
y = Dense(4, activation='softmax')(y)
vgg19_model = Model(inputs=base_model_vgg19.input, outputs=y)

# Compile the models
cnn_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vgg19_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up callbacks with unique filenames
cnn_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_cnn_model.keras'), monitor='val_loss', save_best_only=True)
vgg16_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_vgg16_model.keras'), monitor='val_loss', save_best_only=True)
vgg19_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'best_vgg19_model.keras'), monitor='val_loss', save_best_only=True)

# Train and save the models
cnn_model.fit(train_generator, validation_data=validation_generator, epochs=5, callbacks=[cnn_checkpoint])
cnn_model.save(os.path.join(model_dir, 'cnn_model.keras'))

vgg16_model.fit(train_generator, validation_data=validation_generator, epochs=5, callbacks=[vgg16_checkpoint])
vgg16_model.save(os.path.join(model_dir, 'vgg16_model.keras'))

vgg19_model.fit(train_generator, validation_data=validation_generator, epochs=5, callbacks=[vgg19_checkpoint])
vgg19_model.save(os.path.join(model_dir, 'vgg19_model.keras'))


