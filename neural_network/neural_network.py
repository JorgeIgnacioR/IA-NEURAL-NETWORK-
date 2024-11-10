import numpy as np
import json
import sys
sys.path.append('/Users/jorgeignacioridella/Desktop/PROGRAMACION/Python/Neural-network/layer')


from layer import Layer


class NeuralNetwork:
    def __init__(self) :
        self.layers=[]
        self.loss_list=[] #cada neurona se ha equivocado "x"

    def add_layer(self,num_neuron,input_size):
        if not self.layers:
            self.layers.append(Layer(num_neuron,input_size))
        else:
            previous_output_size=len(self.layers[-1].neurons)
            self.layers.append(Layer(num_neuron,input_size))
    def forward(self,inputs):
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs
    
    def backward(self,loss_gradient,learning_rate):
         for layer in reversed(self.layers):
             loss_gradient=layer.backward(loss_gradient,learning_rate)

    def train(self,X,Y,epochs=1000, learning_rate=0.1): #"x" le damos; "y" respuestas ;epochs=1000 veces que se entrena
        for epoch in range(epochs):
            loss=0
            for i in range (len(X)):
                output=self.forward(X[i])
                loss += np.mean((Y[i]-output)**2)
                loss_gradient= 2* (output-Y[i])
                self.backward(loss_gradient,learning_rate)
            loss /=len(X)
            self.loss_list.append(loss)
            if epoch%100 == 0:
                print(f"epoch: {epoch},loss: {loss}")

    def predict(self,X):
        predictions=[]
        for i in range(len(X)):
            predictions.append(self.forward(X[i]))
        return np.array(predictions)
    
    def save(self,filename="model.json"):
        model_data={
            "layers":[]
        }
        for layer in self.layers:
            layer_data={
                "num_neuron": len(layer.neurons),
                "input_size":len(layer.neurons[0].weight)if layer.neurons else 0,
                "neurons":layer.to_dict()

            }

            model_data["layers"].append(layer_data)

        with open(filename,"w") as f:
            json.dump(model_data,f)
        print(f"model saved in {filename}")

    def load(self,filename="model.json"):
        with open (filename,"r")as f:
            model_data=json.load(f)

        self.layers=[]

        for layer_data in model_data["layers"]:
            num_neurons=layer_data["num_neuron"]
            input_size=layer_data["input_size"]

            new_layer=Layer(num_neurons,input_size)

            new_layer.from_dict(layer_data["neurons"])

            self.layers.append(new_layer)

        print(f"Model loaded from {filename}")




if __name__=="__main__":
    X=np.array([[0.5,8.2,0.1],
                [0.9,8.7,0.3],
                [0.4,0.5,0.8]])
    Y=np.array([[0.3,0.6,0.9]]).T

    nn=NeuralNetwork()

    #nn.add_layer(num_neuron=3,input_size=3)
    #nn.add_layer(num_neuron=3,input_size=3)
    #nn.add_layer(num_neuron=1,input_size=3)

    #nn.train(X,Y, epochs=10000,learning_rate=0.1)

    #nn.save("model1.json")

    nn.load("model1.json")

    predictions=nn.predict(X)
    print(f"predictions: {predictions}")



import matplotlib.pyplot as plt


Y = np.array([0.3, 0.6, 0.9])

plt.plot(Y, label='Valores Reales')
plt.plot(predictions, label='Predicciones')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.legend()
plt.title('Comparación de Valores Reales y Predicciones')
plt.show()
