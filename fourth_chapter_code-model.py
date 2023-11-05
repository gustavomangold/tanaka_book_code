import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np


# tentei usar imagens próprias mas acho que o exemplo com cifar é mais legal
#i = 0
#images, labels = [], []
#while(True):
#	try:
#		images.append(np.asarray(Image.open('images/circulo_' + str(i) + '.png')) / 255.0)
#		labels.append('circulo')
#		images.append(np.asarray(Image.open('images/quadrado_' + str(i) + '.png')) / 255.0)
#		labels.append('quadrado')
#		i += 1
#	except:
#		print('Fim das imagens')
#		break

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# normalizaçao
train_images, test_images = train_images / 255.0, test_images / 255.0


classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(classes[train_labels[i][0]])    
plt.show()

# o shape é o tamanho da imagem ou, em exemplos de spin, o tamanho de um lattice
# as camadas tem tamanhos definidos conforme desejado, mas as inicias precisam
# ter o tamanho do input e do output
# mais tarde, podemos fazer algo como otimização de hiperparâmetros
model = models.Sequential()
model.add(layers.Conv2D(2, (3, 3), activation = 'relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(4, (3, 3), activation = 'relu'))
model.add(layers.MaxPooling2D((2, 2))) 
model.add(layers.Conv2D(4, (3, 3), activation = 'relu'))

#model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)





