import os   
import numpy as np   
import matplotlib.pyplot as plt   
import cv2   
import pandas as pd   
from sklearn.model_selection import train_test_split   
from sklearn.metrics import confusion_matrix   
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten   
   
data_path = "/Users/nivasm/Documents/UTA ASSIGNMENTS/MACHINE LEARNING/ML_PROJECT/leapGestRecog/"   
     
imagepaths = []   
for root, dirs, files in os.walk(data_path, topdown=False):   
  for name in files:   
    path = os.path.join(root, name)   
    if path.endswith("png"):   
      imagepaths.append(path)   
   
X = []   
y = []   
for path in imagepaths:   
  img = cv2.imread(path)   
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   
  img = cv2.resize(img, (320, 120))   
  X.append(img)   
  category = path.split("/")[-2]   
  label = int(category.split("_")[0])   
  y.append(label)   
     
unique_labels = np.unique(y)   
label_mapping = {label: i for i, label in enumerate(unique_labels)}   
y = [label_mapping[label] for label in y]   
     
X = np.array(X, dtype="uint8")   
X = X.reshape(len(imagepaths), 120, 320, 1)   
y = np.array(y)   
   
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)   
   
 
model = Sequential()   
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(120, 320, 1)))   
model.add(MaxPooling2D((2, 2)))   
model.add(Conv2D(64, (3, 3), activation='relu'))   
model.add(MaxPooling2D((2, 2)))   
model.add(Conv2D(64, (3, 3), activation='relu'))   
model.add(MaxPooling2D((2, 2)))   
model.add(Flatten())   
model.add(Dense(128, activation='relu'))   
model.add(Dense(len(unique_labels), activation='softmax'))   
   
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   
  
history = model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=2, validation_data=(X_test, y_test))   
    
model.save('handrecognition_model.h5')   
   
test_loss, test_acc = model.evaluate(X_test, y_test)   
print('Test accuracy: {:2.2f}%'.format(test_acc*100))   
     
plt.plot(history.history['accuracy'])   
plt.plot(history.history['val_accuracy'])   
plt.title('Model accuracy')   
plt.ylabel('Accuracy')   
plt.xlabel('Epoch')   
plt.legend(['Train', 'Test'], loc='upper left')   
plt.show()   
  
plt.plot(history.history['loss'])   
plt.plot(history.history['val_loss'])   
plt.title('Model loss')   
plt.ylabel('Loss')   
plt.xlabel('Epoch')   
plt.legend(['Train', 'Test'], loc='upper left')   
plt.show()   
   
predictions = model.predict(X_test)   
   
y_pred = np.argmax(predictions, axis=1)   

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred),   
      columns=["Predicted " + str(label) for label in unique_labels],   
      index=["Actual " + str(label) for label in unique_labels]) 

print(df_cm) 
  
def validate_9_images(predictions_array, true_label_array, img_array):   
  class_names = [str(label) for label in unique_labels]   
  plt.figure(figsize=(15,5))   
  for i in range(1, 10):   
    prediction = predictions_array[i]   
    true_label = true_label_array[i]   
    img = img_array[i]   
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)   
    plt.subplot(3,3,i)   
    plt.grid(False)   
    plt.xticks([])   
    plt.yticks([])   
    plt.imshow(img, cmap=plt.cm.binary)   
    predicted_label = np.argmax(prediction)   
    if predicted_label == true_label:   
      color = 'blue'   
    else:   
      color = 'red'   
    plt.xlabel("Predicted: {} {:2.0f}% (True: {})".format(class_names[predicted_label],   
                100*np.max(prediction),   
                class_names[true_label]),   
                color=color)   
  plt.show()   
   
validate_9_images(predictions, y_test, X_test)
