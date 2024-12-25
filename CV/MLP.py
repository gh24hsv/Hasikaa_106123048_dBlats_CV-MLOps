import os
import pickle

from skimage.io import imread # type: ignore
from skimage.transform import resize # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.neural_network import MLPClassifier #type: ignore
from sklearn.metrics import accuracy_score # type: ignore




input_dir = '/Users/hsv/Desktop/DB Ind/clf-data'

categories = ['empty','not_empty']

data = []
labels = []

for cat_idx, cat in enumerate(categories):
    files = os.listdir(os.path.join(input_dir,cat))
    for file in files:
        img_path = os.path.join(input_dir, cat, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img.flatten())
        labels.append(cat_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# print(data[0])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)


classifier = MLPClassifier()
params = {'hidden_layer_sizes': [(100,), (50, 50)], 'activation': ['relu', 'tanh']}

grid_search = GridSearchCV(MLPClassifier(max_iter=500), params)
grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)

score = accuracy_score(y_test, y_pred)
print('{}% of samples were correctly classified'.format(str(score * 100)))

#100.0% of samples were correctly classified

#pickle.dump(best_estimator, open('./model.p', 'wb'))