from credit_risks import CreditRisks
import argparse


def run(path, model=None, predict_file=None, predict_data=None):
	model = CreditRisks(path=path)
	result = model.predict([1, 6, 4, 12, 5, 5, 3, 4, 1, 67, 3, 2, 1, 2, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
	print(model.get_score())

if __name__ == '__main__':
	parsing = argparse.ArgumentParser()
	parsing.add_argument('--data', help="Path to dataset")
	parsing.add_argument('--model', help='Load trained model')
	parsing.add_argument('--prediction_file', help='file to data to prediction')
	parsing.add_argument('--predict', help='quick prediction')
	results = parsing.parse_args()
	if results.data == None:
		raise Exception("Parameter data(path to dataset) is empty")
	run(results.data)