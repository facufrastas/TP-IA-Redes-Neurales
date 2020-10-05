import sys
!{sys.executable} -m pip install numpy
import numpy as np
import math
import csv
import pickle
import random

class Neuron():
    
    def __init__(self, positionLayer, outputNeuron=False):
        self.weights = []
        self.inputs = []
        self.output = None
        
        #Pesos actualizados
        self.updatedWeights = []
        #Determina la neurona de salida
        self.outputNeuron = outputNeuron
        #Variable para la actualización del Back Propagation
        self.delta = None
        #Se usa para la actualización del BackPropagation
        self.positionLayer = positionLayer 
        
    #Función que sirve para poder guardar una referencia de las otras neuronas, a esta neurona en particular
    def actualNeuron(self, neurons):        
        self.outputNeurons = neurons
    
    #Función de activación (Sigmoide)
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    #Asignación de pesos aleatorios, considerando cuántas entradas va a tener la neurona
    def initWeights(self, inputNumber):     
        for i in range(inputNumber + 1):
            self.weights.append(random.uniform(0, 1))
        
    def predict(self, row):    
        #Reseteo de los inputs
        self.inputs = []
        #Iteración sobre los pesos y los features
        activation = 0
        for weight, feature in zip(self.weights, row):
            self.inputs.append(feature)
            activation = activation + weight * feature

        self.output = self.sigmoid(activation)
        return self.output
    
    def update_neuron(self):
        #Actualización de los pesos de las neuronas (reemplaaza el peso actual por los que se usan durante el algoritmo de backpropagation)
        self.weights = []
        for newWeight in self.updatedWeights:
            self.weights.append(newWeight)
    
    def calculate_update(self, learningRate, target):
        #Cálculo del nuevo peso para la neurona
        #Primero se calcula el delta dependiendo si es una neurona oculta o una neurona de salida
        if self.outputNeuron:
            self.delta = (self.output - target) * self.output * (1 - self.output)
        else:
            #Cálculo del delta
            delta_sum = 0
            #Se determina a qué pesos contribuye esta neurona en la capa de salida
            curWeightIndex = self.positionLayer 
            for output_neuron in self.outputNeurons:
                delta_sum = delta_sum + (output_neuron.delta * output_neuron.weights[curWeightIndex])

            #Actualización del delta
            self.delta = delta_sum * self.output * (1 - self.output)
            
        #Reseteo de los pesos actualizados
        self.updatedWeights = []
        
        #Iteración y actualización sobre cada peso
        for curWeight, curInput in zip(self.weights, self.inputs):
            gradient = self.delta * curInput
            newWeight = curWeight - learningRate * gradient
            self.updatedWeights.append(newWeight)
            
#Permite la conexión entre las neuronas para poder aplicar el algoritmo Back Propagation
class Layer():
    def __init__(self, neuronsNumber, isOutputLayer = False):
        self.isOutputLayer = isOutputLayer
        self.neurons = []
        #Creación del número de neuronas dado para la capa
        for i in range(neuronsNumber):
            neuron = Neuron(i,  outputNeuron = isOutputLayer)
            self.neurons.append(neuron)
    
    def attach(self, layer):
        #Vinculación de la neurona de esta capa con otra
        for inNeuron in self.neurons:
            inNeuron.actualNeuron(layer.neurons)
            
    def init_layer(self, inputNumber):
        #Inicialización de los pesos de la neurona en la capa
        #Dando el número correcto de inputNumber va a generar el número correcto de pesos
        #Se itera sobre cada neurona y se inicializa su peso correspondiente con la capa anterior
        for neuron in self.neurons:
            neuron.initWeights(inputNumber)
    
    def predict(self, row):
        #Cálculo de la activación de la capa completa
        #Vector aumentado
        row = np.append(row, 1)
        activations = [neuron.predict(row) for neuron in self.neurons]
        return activations

class MultiLayerPerceptron():
    #Creación del perceptrón multicapa con 2 capas: Una de input, una de perceptrons, y una de salida que hace la clasificación binaria
    def __init__(self, learningRate, numIteration):
        self.layers = []
        self.learningRate = learningRate
        self.numIteration = numIteration
        
    def add_output_layer(self, neuronsNumber):
        #Creación de una capa de salida y agregación a la arquitectura
        self.layers.insert(0, Layer(neuronsNumber, isOutputLayer = True))
    
    def add_hidden_layer(self, neuronsNumber):
        #Creación de una capa oculta y agregación al frente de la arquitectura
        hiddenLayer = Layer(neuronsNumber)
        #Agregación de la última capa agregada a esta
        hiddenLayer.attach(self.layers[0])
        #Agregación de esta capa a la arquitectura
        self.layers.insert(0, hiddenLayer)
        
    def update_layers(self, target):
        #Actualización de las capas calculando los nuevos pesos y actualizándolos (todos a la vez) una vez que los nuevos pesos se encontraron
        #Iteración sobre cada una de las capas, en orden inverso, para el cálculo de los pesos
        for layer in reversed(self.layers):
                           
            #Cálculo de la actualización
            for neuron in layer.neurons:
                neuron.calculate_update(self.learningRate, target)  
        
        #Iteración sobre cada una de las capas para la actualización de los pesos
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.update_neuron()
    
    def fit(self, X, y):
        num_row = len(X)
        #Considerando matriz rectangular
        num_feature = len(X[0])
        
        #Inicialización de los pesos en cada una de las capas
        self.layers[0].init_layer(num_feature)
        
        for i in range(1, len(self.layers)):
            inputWeights = len(self.layers[i-1].neurons)
            self.layers[i].init_layer(inputWeights)

        for i in range(self.numIteration):            
            r_i = random.randint(0,num_row - 1)
            #Se toma una fila aleatoria del dataset
            row = X[r_i]
            #Se calcula el error con el método 'predict()'
            yhat = self.predict(row)
            target = y[r_i]
            
            #Actualización de las capas
            self.update_layers(target)
            
            #Cálculo del error cada 100000 iteraciones
            if i % 100000 == 0:
                totalError = 0
                for r_i in range(num_row):
                    row = X[r_i]
                    yhat = self.predict(row)
                    error = (y[r_i] - yhat)
                    totalError = totalError + error ** 2
                mean_error = totalError/num_row
                print(f"Iteración {i} con error = {mean_error}")
        
    
    def predict(self, row):
        activations = self.layers[0].predict(row)
        for i in range(1, len(self.layers)):
            activations = self.layers[i].predict(activations)

        outputs = []
        for activation in activations:                        
            #Considerando si la salida será 0 o 1
            if activation >= 0.5:
                outputs.append(1.0)
            else:
                outputs.append(0.0)
                           
        return outputs[0]

    def save_params(self, nombre="params"):
        pickle.dump(self.weights, open(nombre + ".pickle", "wb"))
        
    def load_params(self, nombre="params"):
        self.weights = pickle.load(open(nombre + ".pickle"), "rb")




# ------------------------------------------------------------------------------
# Lectura de datos
X = np.genfromtxt('X_train.csv', delimiter=',')
Y = np.genfromtxt('Y_train.csv', delimiter=',')
#Preprocesamiento si es necesario...........




# ------------------------------------------------------------------------------
# Clasificación y evaluación
#Inicializando los parámetros
clasif = MultiLayerPerceptron(learningRate = 0.15, numIteration = 1500000)
#Creación de la arquitectura
clasif.add_output_layer(neuronsNumber = 1)
clasif.add_hidden_layer(neuronsNumber = 20)
#Entrenamiento de la red
clasif.fit(X,Y)
#res = clasif.predict(X)

resultados_train = []
for xValue in X:
    resultados_train.append(clasif.predict(xValue))
res = np.array(resultados_train)

#Evaluación del clasificador para los datos de entrenamiento
cant_datos = X.shape[0]
accuracy = (cant_datos - abs(res - Y).sum())/cant_datos
print("Precisión:", accuracy)

#Evaluación con datos de test (no cátedra)....




# ------------------------------------------------------------------------------
#Validación de la cátedra y exportación de resultados y parámetros
X_test = np.genfromtxt('X_test.csv', delimiter=',')
#Preprocesamiento si es necesario...........
#res_test = clasif.predict(X_test)

resultados = []
for xValue in X_test:
    resultados.append(clasif.predict(xValue))
res_test = np.array(resultados)

np.savetxt('Y_test_Grupo_01.csv', res_test, delimiter=',')
#clasif.save_params("params_Grupo_01")