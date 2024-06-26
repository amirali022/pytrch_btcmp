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