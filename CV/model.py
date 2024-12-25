import os
import pickle

from skimage.io import imread # type: ignore
from skimage.transform import resize # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.svm import SVC # type: ignore
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

# SVC is a pretty simple model so we train 3 x 4 models having different calues of gamma and C 
# then choose the best among them using GridSearchCV and best_estimator_
classifier = SVC()
params = [{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]

grid_search = GridSearchCV(classifier, params)
grid_search.fit(x_train, y_train)

best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(x_test)

score = accuracy_score(y_test, y_pred)
print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb'))

#99.9178% of samples were correctly classified