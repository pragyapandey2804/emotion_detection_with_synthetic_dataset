from img2vec_pytorch import Img2Vec
from PIL import Image
import pickle

with open('./model.p', 'rb') as f:
    model = pickle.load(f)

img2Vec = Img2Vec()

image_path ="./content/faces_dataset/val/angry/0214.png"
img = Image.open(image_path)

features = img2Vec.get_vec(img)

pred = model.predict([features])
print(pred)