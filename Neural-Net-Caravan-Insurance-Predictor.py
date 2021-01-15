import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.utils.vis_utils import plot_model
import matplotlib.style as style


ds = pd.read_csv('coil2000feat.csv',header=None)
ds = ds.reindex(np.random.permutation(ds.index))
train_features = ds
train_label = pd.read_csv('coil2000label.csv',header=None)


s = StandardScaler()
for x in train_features.columns:
        train_features[x] = s.fit_transform(train_features[x].values.reshape(-1, 1)).astype('float64')


input_dim = train_features.shape[1]
model = keras.models.Sequential()
model.add(keras.layers.Dense(32, input_dim = input_dim, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(16, activation=tf.keras.layers.LeakyReLU(),kernel_initializer='he_uniform'))
model.add(keras.layers.Dense(1, activation='sigmoid',kernel_initializer='he_uniform'))


model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              metrics = 'binary_accuracy')


history = model.fit(train_features, train_label, epochs=60, validation_split=0.5)

metrics = np.mean(history.history['val_binary_accuracy'])
results = model.evaluate(train_features, train_label)
print('\nLoss, Binary_accuracy: \n',(results))


style.use('dark_background')
pd.DataFrame(history.history).plot(figsize=(11, 7),linewidth=4)
plt.title('Binary Cross-entropy',fontsize=14, fontweight='bold')
plt.xlabel('Epochs',fontsize=13)
plt.ylabel('Metrics',fontsize=13)
plt.show()