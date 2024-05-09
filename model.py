from torch import nn
import torch.nn.functional as F

from config import *

class EmotionDetectorModel(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
		self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
		self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		# the window size is 200, then after using max pooling it's 50
		new_size = NN_IMAGE_SIZE // 4

		self.fc1 = nn.Linear(32 * new_size * new_size, 128)
		self.fc2 = nn.Linear(128, NR_OF_CATEGORIES)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.max_pool1(x)

		x = F.relu(self.conv2(x))
		x = self.max_pool2(x)

		# flatten the output for the fully connected layers
  
		new_size = NN_IMAGE_SIZE // 4
		x = x.view(-1, 32 * new_size * new_size)

		x = F.relu(self.fc1(x))
		x = F.softmax(self.fc2(x), dim=1)

		return x