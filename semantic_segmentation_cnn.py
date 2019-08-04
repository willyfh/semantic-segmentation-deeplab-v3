from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import numpy as np


# pretained model, deeplabv3_resnet101
model = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()

img = Image.open('./sample_image.png')

# apply the transformations needed
transform = T.Compose([T.Resize(256),
				T.CenterCrop(224),
				T.ToTensor(), 
				T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
inp = transform(img).unsqueeze(0)


# pass the input through the net, Here the output shape is (1, 21, 24, 24)
out = model(inp)['out']

# convert to shape (224, 224)
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()

# define the helper function
def decode_segmap(image, nc=21):
	label_colors = np.array([(0, 0, 0),  # 0=background
						# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
						(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
						# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
						(0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
						# 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
						(192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
						# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
						(0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

	r = np.zeros_like(image).astype(np.uint8)
	g = np.zeros_like(image).astype(np.uint8)
	b = np.zeros_like(image).astype(np.uint8)

	for l in range(0, nc):
		idxs = image == l
		r[idxs] = label_colors[l, 0]
		g[idxs] = label_colors[l, 1]
		b[idxs] = label_colors[l, 2]

	rgb = np.stack([r, g, b], axis=2)
	return rgb

# test	
rgb = decode_segmap(om)
plt.imshow(rgb)
plt.show()