import tensorflow as tf
import tensorflow_datasets as tfds
from keras._tf_keras.keras import Model, Input
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

dataset, info = tfds.load(
    'voc/2007',
    split=['train', 'validation'],
    with_info=True
)

class_names = info.features['objects']['label'].names
num_classes = len(class_names)
img_size = (128, 128)

def random_flip_horizontal(image, bbox):
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        ymin, xmin, ymax, xmax = tf.unstack(bbox)
        xmin, xmax = 1.0 - xmax, 1.0 - xmin
        bbox = tf.stack([ymin, xmin, ymax, xmax])
    return image, bbox

def random_brightness(image, max_delta=0.1):
    image = tf.image.random_brightness(image, max_delta)
    return image

def augment(image, bbox):
    image, bbox = random_flip_horizontal(image, bbox)
    image = random_brightness(image)
    return image, bbox

def preprocess(sample, training=False):
    image = tf.image.resize(sample['image'], img_size) / 255.0
    bbox = sample['objects']['bbox'][0]
    label = tf.cast(sample['objects']['label'][0], tf.float32)
    if training:
        image, bbox = augment(image, bbox)
    return image, {'bbox': bbox, 'label': label}

train_data = dataset[0].map(lambda x: preprocess(x, training=True)).batch(32).prefetch(tf.data.AUTOTUNE)
val_data = dataset[1].map(lambda x: preprocess(x, training=False)).batch(32).prefetch(tf.data.AUTOTUNE)

inputs = Input(shape=img_size + (3,))
x = Conv2D(32, (3,3), activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(64, (3,3), activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)

bbox_output = Dense(4, activation='sigmoid', name='bbox')(x)
class_output = Dense(num_classes, activation='softmax', name='label')(x)

model = Model(inputs=inputs, outputs={'bbox': bbox_output, 'label': class_output})

model.compile(
    optimizer=Adam(),
    loss={
        'bbox': 'mse',
        'label': 'sparse_categorical_crossentropy'
    },
    metrics={
        'bbox': 'mse',
        'label': 'accuracy'
    }
)

model.summary()

model.fit(train_data, validation_data=val_data, epochs=20)
model.save("detc.keras")



def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    return img, img_array

def predict_on_image(img_array):
    img_tensor = tf.expand_dims(img_array, axis=0)  
    preds = model.predict(img_tensor)
    bbox = preds['bbox'][0].numpy()
    label = preds['label'][0].numpy()
    class_id = np.argmax(label)
    class_name = class_names[class_id]
    return bbox, class_name

def draw_bbox_on_image(img, bbox, class_name):

    w, h = img.size
    ymin, xmin, ymax, xmax = bbox
    left = int(xmin * w)
    right = int(xmax * w)
    top = int(ymin * h)
    bottom = int(ymax * h)

    img_draw = img.copy()
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([left, top, right, bottom], outline='red', width=3)
    draw.text((left, top-15), class_name, fill='red')
    return img_draw

root = tk.Tk()
root.title("Détection d'objets simplifiée - VOC2007")
root.geometry("400x500")

label_img = tk.Label(root)
label_img.pack()

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not filepath:
        return
    original_img, preproc_img = load_and_preprocess_image(filepath)
    bbox, class_name = predict_on_image(preproc_img)
    img_with_box = draw_bbox_on_image(original_img, bbox, class_name)

    img_tk = ImageTk.PhotoImage(img_with_box.resize((350, 350)))
    label_img.config(image=img_tk)
    label_img.image = img_tk
    label_img.pack()

btn = tk.Button(root, text="Charger une image", command=open_file)
btn.pack()

root.mainloop()
