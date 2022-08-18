from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("vegetable.h5")
img = image.image_utils.load_img('tsls1.JPG', target_size = (64, 64))
x = image.image_utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
print(np.argmax(pred, axis=1))
class_names = ["Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Tomato___Bacterial_spot", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot"]
print(class_names[int(np.argmax(pred, axis=1))])



