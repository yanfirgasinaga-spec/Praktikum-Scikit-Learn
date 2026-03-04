import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data_dict = pickle.load(open(r"D:\Semester 6\Kontrol Cerdas\Praktikum 3\DATA.pickle", "rb"))

clean_data = []
clean_labels = []

for d, lbl in zip(data_dict['data'], data_dict['labels']):
    arr = np.array(d)
    if arr.shape == (42,):   
        clean_data.append(arr)
        clean_labels.append(lbl)

clean_data = np.array(clean_data)
clean_labels = np.array(clean_labels)

print("Data bersih:", clean_data.shape)
print("Label bersih:", clean_labels.shape)


x_train, x_test, y_train, y_test = train_test_split(
    clean_data, clean_labels, test_size=0.2, shuffle=True, stratify=clean_labels
)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")


with open(r"D:\Semester 6\Kontrol Cerdas\Praktikum 3\model.p", "wb") as f:
    pickle.dump(model, f)