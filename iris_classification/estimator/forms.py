from django import forms

class PredictionForm(forms.Form):
	sepal_length = forms.CharField(label='Sepal Length')
	sepal_width = forms.CharField(label='Sepal Width')
	petal_length = forms.CharField(label='Petal Length')
	petal_width = forms.CharField(label='Petal Width')