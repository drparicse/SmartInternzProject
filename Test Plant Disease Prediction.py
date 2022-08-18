from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("fruit.h5")
img = image.image_utils.load_img('abr1.JPG', target_size = (64, 64))
x = image.image_utils.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = model.predict(x)
print(np.argmax(pred, axis=1))
class_names = ["Apple___Black_rot", "Apple___healthy", "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Peach___Bacterial_spot", "Peach___healthy"]
print(class_names[int(np.argmax(pred, axis=1))])



