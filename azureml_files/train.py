import sklearn
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import makedirs
from azureml.core import Model, Run, Dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

def get_parsed_args():
	parser = argparse.ArgumentParser(description='Iris dataset classification parameters')
	parser.add_argument('--model-name', type=str, help='Name for the trained model')
	parser.add_argument('--dataset-name', type=str, help='Name of the dataset')
	parser.add_argument('--test-size', type=float, help='Test size in the train test split')
	parser.add_argument('--random-state-test', type=int, help='Random state in the train test split')
	parser.add_argument('--random-state-model', type=int, help='Random state in the model')
	parser.add_argument('--max-leaf-nodes', type=int, help='Max leaf nodes')
	parser.add_argument('--output-folder', type=str, help='Output folder')
	return parser.parse_args()

def log_arguments(args, dataset, run):
	run.log('dataset_name', dataset.name)
	run.log('dataset_version', dataset.version)
	run.log('test_size', args.test_size)
	run.log('random_state_test', args.random_state_test)
	run.log('random_state_model', args.random_state_model)
	run.log('max_leaf_nodes', args.max_leaf_nodes)

def log_results(model_accuracy, output_folder, model_name, matrix, labels, run):
	run.log('model_accuracy', model_accuracy)
	run.log_image('Confusion Matrix Plot', plot=create_confusion_matrix_plot(matrix, labels))
	model = Model.register(
		workspace=run.experiment.workspace,
		model_path=output_folder,
		model_name=model_name,
		model_framework=Model.Framework.SCIKITLEARN,
		model_framework_version=sklearn.__version__,
		tags={'Training context': 'Pipeline'}
	)
	run.log('model_name', model.name)
	run.log('model_version', model.version)

def create_confusion_matrix_plot(matrix, labels):
	normalized_matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
	plt.figure(figsize=(8, 6))
	plt.imshow(normalized_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
	plt.title('Confusion Matrix')
	plt.colorbar()
	tick_marks = np.arange(len(labels))
	plt.xticks(tick_marks, labels)
	plt.yticks(tick_marks, labels)
	return plt

def train_decision_tree(max_leaf_nodes, random_state, X_train, y_train):
	iris_classifier = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=random_state)
	iris_classifier.fit(X_train, y_train)
	return iris_classifier

def predict(model, X_test):
	y_prediction = model.predict(X_test)
	return y_prediction

def save_model(model, model_name, output_folder):
	output_path = output_folder + '/{}.pkl'.format(model_name)
	joblib.dump(value=model, filename=output_path)

def main(args, run, pd_dataset):
	logging.basicConfig(level=logging.INFO)
	pd_dataset = dataset.to_pandas_dataframe()
	X, y = pd_dataset.iloc[:, :-1], pd_dataset.iloc[:, -1]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state_test)
	model = train_decision_tree(args.max_leaf_nodes, args.random_state_model, X_train, y_train)
	y_prediction = predict(model, X_test)
	model_accuracy = accuracy_score(y_true=y_test, y_pred=y_prediction)
	labels = y.unique()
	matrix = confusion_matrix(y_test, y_prediction, labels)
	save_model(model, args.model_name, args.output_folder)
	log_results(model_accuracy, args.output_folder, args.model_name, matrix, labels, run)

if __name__ == '__main__':
	run = Run.get_context()
	args = get_parsed_args()
	makedirs(args.output_folder, exist_ok=True)
	dataset = Dataset.get_by_name(run.experiment.workspace, name=args.dataset_name)
	log_arguments(args, dataset, run)
	main(args, run, dataset)
	run.complete()