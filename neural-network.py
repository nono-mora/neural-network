#Training data. 
inputs = [1, 2, 3, 4]
targets = [2, 4, 6, 8]

# "w" is our slope/weight. 
w = 0.1
epochs = 10
learning_rate = 0.1

def predict(i):
    return w * i

#Train the network
for _ in range(epochs):
    pred = [predict(i) for i in inputs] #Calculates a prediction for al the inputs. 
    errors = [(p - t) ** 2 for p, t in zip(pred, targets)] #Calculates the error between the prediction and the target.
    cost = sum(errors)/len(targets) #Calculates the average of the errors values. 
    print(f"Weight: {w:.2f}, Cost: {cost:.2f}")
    
    #Back Propagation 
    errors_derivative = [2 * (p - t) for p, t in zip(pred, targets)] #Calculates the derivative of the errors.
    weight_delta = [e * i for e, i in zip(errors_derivative, inputs)] #Calculates weight delta.

    w -= learning_rate * sum(weight_delta) / len(weight_delta) #Updates the weight. If the cost reaches 0, the weight will be the best possible value and learning will stop. 

#Test neural network
result = predict(15)
print(f"{result:.2f}")