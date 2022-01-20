import torchvision

import models
import datasets
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from matplotlib import pyplot as plt

IMAGES_TO_USE = 1000
CONTENT_CODE_LEN = 20
BATCH_SIZE = 16


def train_model(model, tboard_name, epochs=50, lr=1e-3, noise_std=0.3, reg_factor=1e-6):
	writer = SummaryWriter(log_dir='logs/' + tboard_name)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	l1_loss = torch.nn.L1Loss(reduction='mean')
	l2_loss = torch.nn.MSELoss()
	loss_func = lambda x, y, c: l1_loss(x, y) + l2_loss(x, y) + reg_factor * torch.norm(c) ** 2
 
    # prepare the data
	data = datasets.MNIST(IMAGES_TO_USE)
	train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=2)
	class_codes = torch.eye(10)
	content_codes = torch.normal(0, noise_std, (IMAGES_TO_USE, CONTENT_CODE_LEN))
    
    # set up some variables for the visualizations
	display_contents = [100, 150, 200, 300, 400]
	display_classes = [6, 9, 3, 4, 2]
	tboard_contents = torch.cat(((torch.zeros(1, 1, 28, 28), torch.Tensor([data[i][0] for i in display_contents]))))

	for epoch in range(epochs):
		model.train()

		losses = []
		for data_row in train_loader:
			# get the inputs; data is a list of [inputs, labels]
			images, labels, indices = data_row

			# create input for network
			cur_content_codes = content_codes[indices]
			cur_content_codes.requires_grad_(True)
			noisy_code = cur_content_codes + torch.rand(CONTENT_CODE_LEN) * noise_std
			inputs = torch.cat((class_codes[labels], noisy_code), 1)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)

			loss = loss_func(outputs, images, cur_content_codes)
			loss.backward()
			optimizer.step()

			# statistics
			losses.append(loss.item())

		model.eval()
        out_batch = model(torch.cat((batch_codes, batch_contents), 1))
        # format the display
		batch_codes = class_codes[np.tile(display_classes, len(display_contents))]
		batch_contents = content_codes[np.repeat(display_contents, len(display_classes))]
		tboard_batch = torch.zeros((len(display_classes)+1)**2, 1, 28, 28)
		first_column = np.arange(0, tboard_batch.shape[0], len(display_classes)+1)
		non_first_col = np.arange((len(display_classes)+1)**2)
		non_first_col = non_first_col[non_first_col % 6 != 0]
		tboard_batch[first_column, ...] = tboard_contents
		tboard_batch[non_first_col, ...] = torch.cat((tboard_contents[1:], out_batch))
        
        # add to tensorboard
		grid = torchvision.utils.make_grid(tboard_batch, nrow=len(display_classes)+1)
		writer.add_image('images', grid, epoch)
		writer.add_scalar('loss', np.mean(losses), epoch)
		print("Epoch: {}, loss: {}\n".format(epoch, np.mean(losses)))

	writer.close()

	return model


if __name__ == '__main__':
	model = models.GeneratorForMnistGLO(code_dim=CONTENT_CODE_LEN + 10)
	model = train_model(model, "100_epochs", epochs=100, lr=2e-3, reg_factor=1e-6)
	print("Done")
