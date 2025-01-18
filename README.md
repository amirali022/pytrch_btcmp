## PyTorch for Deep Learning Bootcamp

### Requirements

- Python 3.11
- Jupyter Notebook

### Libraries

- numpy
- pandas
- matplotlib
- torch + cuda

### Notebooks

1. Fundamentals

	- [PyTorch Fundamentals](/notebooks/01_Fundamentals/01_pytorch_fundamentals.ipynb)

		- working with tensors

2. Workflow

	- [PyTorch Workflow](/notebooks/02_Workflow/01_pytorch_workflow.ipynb)

		- build a simple linear regression model
		- training and testing
		- saving and loading model

	- [PyTorch Workflow 2](/notebooks/02_Workflow/02_pytorch_workflow_device_agnostic.ipynb)

		- build a device agnostic model
		- using `nn.Linear` for linear regression task

3. Neural Network Classification

	- [PyTorch Classification](/notebooks/03_Classification/01_pytorch_classification.ipynb)

		- create toy dataset for binary classification
		- create linear and non-linear models
		- plot decision boundary

	- [Multi-Class Classification](/notebooks/03_Classification/02_multiclass_classification.ipynb)

		- create toy dataset for muti-class classification
		- create model for multi-class classification

	- [Classification Exercises](/notebooks/03_Classification/03_pytorch_classification_exercises.ipynb)

		- **Binary classification:**
			- Dataset: Scikit-learn's `make_moons()`
			- Model: Feedforward non-linear (ReLU) Neural Network

		- **Multi-Class Classification:**
			- Dataset: 3 class spirals
			- Model: Feedforward non-linear (Tanh) Neural Network

4. Computer Vision

	- [PyTorch Computer Vision](/notebooks/04_Vision/01_pytorch_computer_vision.ipynb)

		- Work with FashionMNIST Dataset with `torch.utils.data.DataLoader`
		- Build, train and evaluate baseline feedforward neural network with and with non-linearity
		- Functionizing training_step, testing_step and model evaluation
		- Build, train and evaluate Convolutional Neural Network (CNN) which replicates TinyVGG architecture
		- Study `torch.nn.Conv2d` and `torch.nn.MaxPool2d`
		- Compare results of all evaluated models
		- Compute and visulize confusion matrix
		- Save and Load best performing model

	- [PyTorch Computer Vision Excercise](/notebooks/04_Vision/02_pytorch_computer_vision_exercises.ipynb)

		- Work with MNIST dataset
		- Train TinyVGG model on MNIST dataset using cpu and gpu
		- Functionizing fit function
		- Plotting loss and accuracy curve

5. Custom Datasets

	- [PyTorch Custom Datasets](/notebooks/05_Custom_Datasets/01_pytorch_custom_datasets.ipynb)

		- Work with pizza_steak_sushi dataset
		- Loading Data using ImageFolder
		- Loading Data using Custom Dataset
		- Data Transformation + Data Augmentation
		- Fitting TinyVGG Model on Dataset with and without Transform
		- Evaluating and Comparing results using Loss and Accuracy Curves

6. Going Modular

	- [Going Modular (cell mode)](/notebooks/06_Going_Modular/01_pytorch_going_modular.ipynb)

		- a condensed version of the previous chapter's notebook [PyTorch Custom Datasets](/notebooks/05_Custom_Datasets/01_pytorch_custom_datasets.ipynb)
		- containing `%%writefile` magic function to create scripts of different parts of notebook with different functionality:
			- [data_setup.py](/notebooks/06_Going_Modular/going_modular/data_setup.py): create train/test DataLoaders
			- [model_builder.py](/notebooks/06_Going_Modular/going_modular/model_builder.py): instantiate the PyTorch model
			- [engine.py](/notebooks/06_Going_Modular/going_modular/engine.py): training and testing functions
			- [utils.py](/notebooks/06_Going_Modular/going_modular/utils.py): utility function for saving PyTorch model
			- [train.py](/notebooks/06_Going_Modular/going_modular/train.py): main function for training a model with custom hyperparameters using `argparse`
			- [get_data.py](/notebooks/06_Going_Modular/going_modular/get_data.py): script for downloading the dataset

7. Transfer Learning

	- [Transfer Learning](/notebooks/07_Transfer_Learning/01_pytorch_transfer_learning.ipynb)
		- working with EfficientNet_B0 model trained on imagenet
		- prepare data transformation according to pre-trained model (manual, auto)
		- freezing feature layers and updating classifier layer
		- training and evaluate the training process using loss curves