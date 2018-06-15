Language Used: Python 3.6

Pickle persistence library used to store the trained model weights and then use trained model for prediction.

To Execute from command line: python nn.py training-data-filename activation-function-option

	2 parameters needed : 

			training dtatset -> iris.csv/adult.data/car/data
			activation-function-option -> 1:Sigmoid 2:tanh 3:relu

e.g.: 

	python nn.py adult.data 1
	python nn.py adult.data 2
	python nn.py adult.data 3
