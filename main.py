from img2vec_pytorch import Img2Vec
import os
import pickle
from PIL import Image 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score

##data preparation

img2vec = Img2Vec()

data_dir = "./content/faces_dataset"
train_dir = os.path.join(data_dir, "training")
val_dir = os.path.join(data_dir,"val")

data= {}
for j, dir_ in enumerate([train_dir,val_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_,category)):
            img_path_ = os.path.join(dir_,category,img_path)
            img = Image.open(img_path_)

            if img.mode !='RGB':
                img = img.convert('RGB')

            img_features = img2vec.get_vec(img)
            features.append(img_features)
            labels.append(category)

    data[['training_data','validation_data'][j]] = features
    data[['training_labels','validation_labels'][j]] = labels      

#model training

model = RandomForestClassifier()
model.fit(data['training_data'], data['training_labels'])

##performance test

y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])
print(score)

##dowmloading the model

with open('./model.p','wb') as f:
    pickle.dump(model, f)
    f.close()