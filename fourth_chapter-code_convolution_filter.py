import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from PIL import Image

image = np.asarray(Image.open('population.png').convert('L'))

def show_image(image):
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(image, cmap = 'gray', vmin = 0, vmax = 255)    
	plt.show()

filter_ = [[0, 0, 0, 0],
					 [0, 0.5, 0.3, 0],
					 [-0, 0.5, 0.2, 0],
					 [0, 0, 0, 0]]
					 
m, n = len(filter_), len(filter_)

show_image(image)

image_resized = np.lib.pad(image, ((m-1,m-1),(n-1,n-1)), 'constant', constant_values=(0)) 
i=0
for index in range(1, len(image)-2):
	new_pixel = 0
	for subindex in range(1, len(image[0])-2):
		submatrix = []
		for innerindex in range(m):
			subline = []
			for subinnerindex in range(n):
				subline.append(image[index+innerindex-1][subindex+subinnerindex-1])
			submatrix.append(subline)
			
		new_pixel += filter_*np.array(submatrix)
		image[index][subindex] = int(sum(sum(new_pixel))/255)

show_image(image)
