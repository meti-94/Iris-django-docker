import numpy as np
from django.shortcuts import render
from .forms import PredictionForm
from .classifier import make_prediction
from django.http import HttpResponse
import json
from django.http import JsonResponse
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
from rest_framework.decorators import api_view
from .serializers import InputJsonSerializer, OutputJsonSerializer


def home(request):
	return render(request, 'estimator/home.html')

def dataset(request):
	return render(request, 'estimator/dataset.html', {'title': 'Dataset'})

def predict(request):
	if request.method == 'POST':
		form = PredictionForm(request.POST)
		if form.is_valid():
			pred_sepal_length = float(form.cleaned_data['sepal_length'])
			pred_sepal_width = float(form.cleaned_data['sepal_width'])
			pred_petal_length = float(form.cleaned_data['petal_length'])
			pred_petal_width = float(form.cleaned_data['petal_width'])
			arr = [float(form.cleaned_data[key]) for key in form.cleaned_data.keys()]
			arr = np.array(arr).reshape(1, -1)
			prediction = make_prediction(arr)
			print("Prediction: ", prediction)
	else:
		form = PredictionForm()
		prediction = ''
	return render(request, 'estimator/predict.html', {'form': form, 'prediction': prediction})




@api_view(['POST'])
def api(request):
	serialized_request = InputJsonSerializer(data=request.data)
	if serialized_request.is_valid():
		request_data = serialized_request.validated_data
		pred_sepal_length = request_data['sepal_length']
		pred_sepal_width = request_data['sepal_width']
		pred_petal_length = request_data['petal_length']
		pred_petal_width = request_data['petal_width']
		arr = np.array([pred_sepal_length,
					pred_sepal_width,
					pred_petal_length,
					pred_sepal_width]).reshape(1, -1)
		_response = {"Prediction":make_prediction(arr)}
		print(_response)
		output_serializer = OutputJsonSerializer(_response)
		return Response(output_serializer.data)
	else:
		return Response(serialized_request.errors)
