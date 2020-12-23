from iris_classification.settings import BASE_DIR
import pickle
import os


def make_prediction(X):
	pth = os.path.join(BASE_DIR, 'estimator/classifier/model.sav')
	print(pth)
	model = pickle.load(open(pth, 'rb'))

	prediction = model.predict(X)
	if prediction == 0:
		return "Iris Setosa"
	elif prediction == 1:
		return "Iris Virginica"
	else:
		return "Iris Versicolor"