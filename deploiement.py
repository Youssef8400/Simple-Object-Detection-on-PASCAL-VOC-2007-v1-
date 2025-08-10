import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

model_path = "detc.keras"
img_size = (128, 128)

model = tf.keras.models.load_model(model_path)

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    return img, img_array

def predict_on_image(img_array):
    img_tensor = tf.expand_dims(img_array, axis=0)
    preds = model.predict(img_tensor)
    bbox = preds['bbox'][0]
    return bbox  # 🔹 Retourne seulement la bounding box

def draw_bbox_on_image(img, bbox):
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
    return img_draw

root = tk.Tk()
root.title("Tester modèle détection")

label_img = tk.Label(root)
label_img.pack()

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not filepath:
        return
    original_img, preproc_img = load_and_preprocess_image(filepath)
    bbox = predict_on_image(preproc_img)
    img_with_box = draw_bbox_on_image(original_img, bbox)

    img_tk = ImageTk.PhotoImage(img_with_box.resize((350, 350)))
    label_img.config(image=img_tk)
    label_img.image = img_tk
    label_img.pack()

btn = tk.Button(root, text="Charger une image", command=open_file)
btn.pack()

root.mainloop()
