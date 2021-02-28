from rest_framework import serializers



class InputJsonSerializer(serializers.Serializer):
    sepal_length = serializers.FloatField(max_value=None, min_value=0)
    sepal_width = serializers.FloatField(max_value=None, min_value=0)
    petal_length = serializers.FloatField(max_value=None, min_value=0)
    petal_width = serializers.FloatField(max_value=None, min_value=0)


class OutputJsonSerializer(serializers.Serializer):
    Prediction = serializers.ChoiceField(['Iris Versicolor', 'Iris Setosa', 'Iris Virginica'])