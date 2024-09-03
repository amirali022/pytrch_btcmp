"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
from torchvision import transforms
from timeit import default_timer as timer
import data_setup, engine, model_builder, utils
import argparse
from datetime import datetime

if __name__ == "__main__":

	parser = argparse.ArgumentParser( description="script for training a TinyVGG model on pizza steak sushi dataset")
	parser.add_argument( "--learning_rate", type=float, default=0.001, required=False, help="specify learning rate (e.g. 0.001)")
	parser.add_argument( "--batch_size", type=int, default=32, required=False, help="specify batch size for loading data (e.g. 32)")
	parser.add_argument( "--num_epochs", type=int, default=5, required=False, help="specify the number of epochs (e.g. 5)")
	parser.add_argument( "--hidden_units", type=int, default=10, required=False, help="specify the number of hidden units/filters in model (e.g. 10)")

	args = parser.parse_args()

	# # Setup hyperparameters
	NUM_EPOCHS = args.num_epochs
	BATCH_SIZE = args.batch_size
	HIDDEN_UNITS = args.hidden_units
	LEARNING_RATE = args.learning_rate

	print( "Hyper Parameters:")
	print( f"Number of Epochs: { NUM_EPOCHS}")
	print( f"Batch Size: { BATCH_SIZE}")
	print( f"Number of Hidden Units: { HIDDEN_UNITS}")
	print( f"Learning Rate: { LEARNING_RATE}")

	# Setup directories
	train_dir = "data/pizza_steak_sushi/train"
	test_dir = "data/pizza_steak_sushi/test"

	# Setup device-agnostic code
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Create transforms
	data_transform = transforms.Compose( [
		transforms.Resize( ( 64, 64)),
		transforms.ToTensor()
	])

	# Create DataLoaders and get class_names
	train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders( train_dir=train_dir,
																					test_dir=test_dir,
																					transform=data_transform,
																					batch_size=BATCH_SIZE,
																					num_workers=1)

	# Create model
	model = model_builder.TinyVGG( input_shape=3,
								hidden_units=HIDDEN_UNITS,
								output_shape=len( class_names)).to( device)

	# Setip loss and optimizer
	loss_fn = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam( model.parameters(), lr=LEARNING_RATE)


	# Start the timer
	start_time = timer()

	# Start training with help from engine.py
	engine.train( model=model,
				train_dataloader=train_dataloader,
				test_dataloader=test_dataloader,
				loss_fn=loss_fn,
				optimizer=optimizer,
				epochs=NUM_EPOCHS,
				device=device)


	# End the timer and print out how long it took
	end_time = timer()

	print( f"[INFO] Total training time: { end_time-start_time:.3f} seconds")

	# Save the model to file
	utils.save_model( model=model,
					target_dir="models",
					model_name=f"01_going_modular_script_mode_tinyvgg_model_{ datetime.now().isoformat()}.pth")
