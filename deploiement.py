import tensorflow as tf
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import filedialog

MODEL_PATH = "detc.keras"
IMG_SIZE = (128, 128)
WINDOW_SIZE = (500, 500)  
IMG_DISPLAY_SIZE = (350, 350)  

model = tf.keras.models.load_model(MODEL_PATH)

def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    return img, img_array

def predict_on_image(img_array):
    img_tensor = tf.expand_dims(img_array, axis=0)
    preds = model.predict(img_tensor)
    bbox = preds['bbox'][0]  
    return bbox

def draw_bbox_on_image(img, bbox):
    w, h = img.size
    ymin, xmin, ymax, xmax = bbox
    left = int(xmin * w)
    right = int(xmax * w)
    top = int(ymin * h)
    bottom = int(ymax * h)

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle([left, top, right, bottom], outline='red', width=3)
    return img_draw

root = tk.Tk()
root.title("Détection Objet - Testeur")
root.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")
root.resizable(False, False)  

title_label = tk.Label(root, text="Test de Détection", font=("Arial", 18, "bold"))
title_label.pack(pady=10)

frame_img = tk.Frame(root, width=IMG_DISPLAY_SIZE[0], height=IMG_DISPLAY_SIZE[1], bg="#e0e0e0", relief="sunken", bd=2)
frame_img.pack(pady=10)
frame_img.pack_propagate(False)

label_img = tk.Label(frame_img, bg="#e0e0e0")
label_img.pack(expand=True)

def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
    if not filepath:
        return
    original_img, preproc_img = load_and_preprocess_image(filepath)
    bbox = predict_on_image(preproc_img)
    img_with_box = draw_bbox_on_image(original_img, bbox)

    img_tk = ImageTk.PhotoImage(img_with_box.resize(IMG_DISPLAY_SIZE))
    label_img.config(image=img_tk)
    label_img.image = img_tk

btn = tk.Button(root, text="📂 Charger une image", font=("Arial", 12), command=open_file)
btn.pack(pady=15)

root.mainloop()
