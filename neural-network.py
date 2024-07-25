#Training data. The inputs in this case represent how old a car is in years. The target represent the maintenance cost in dollars. The older the car, the higher the maintenance cost.
# We are going to train our neural network to predict the maintenance cost of a car based on its age.
inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]


w = 0.1 # "w" is our slope/weight. 
b = 0.3 # "b" is our bias or y-intercept. 
epochs = 1000 #The number of times we will train our neural network.
learning_rate = 0.05

def predict(i):
    return w * i + b

#Train the network
for epoch in range(epochs):
    pred = [predict(i) for i in inputs] #Calculates a prediction for al the inputs. 
    cost = sum([(p - t) ** 2 for p, t in zip(pred, targets)]) / len(targets) #Calculates the cost.
    print(f"Weight: {w:.2f}, Bias: {b:.2f}, Cost: {cost:.2f}")
    
    #Backpropagation 
    errors_derivative = [2 * (p - t) for p, t in zip(pred, targets)] #Calculates the derivative of the errors.
    weight_delta = [e * i for e, i in zip(errors_derivative, inputs)] #Calculates weight delta.
    bias_delta = [e * 1 for e in errors_derivative] #Calculates bias delta.

    w -= learning_rate * sum(weight_delta) / len(weight_delta) #Updates the weight. 
    b -= learning_rate * sum(bias_delta) / len(bias_delta) #Updates the bias. 

#Test neural network
test_inputs = [1, 2, 4, 5]
test_targets = [555, 1140, 1750, 1790]
pred = [predict(i) for i in test_inputs]
for i , t,  p in zip(test_inputs, test_targets, pred):
    print(f"Input: {i}, Target: {t}, Prediction: {p:.2f}")