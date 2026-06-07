import numpy as np

#Great Credit (0.8), great income,(80K a year) low debt (0.2)
app1 = np.array([0.8, 0.8, 0.2])
y1 = 1 #approved

app2 = np.array ([0.3, 0.4, 0.7])
y2 = -1 #denied

app3 = np.array ([0.6, 0.6, 0.8])
y3 = -1 #denied

app4 = np.array ([0.9, 0.5,0.1])
y4 = 1 #approved

training_data = [(app1, y1), (app2,y2), (app3,y3), (app4,y4)]

print (training_data)

weights = np.array([0.0, 0.0, 0.0])

bias = 0.0
#in a realistic scenario, scientists adjust the the learning rate to find the best fit for the data. A learning rate that is too high may cause the model to converge too quickly to a suboptimal solution, while a learning rate that is too low may result in a long training time and may get stuck in local minima.
#learning rate means how much the weights and bias are updated during each iteration of training. It determines the step size at which the model learns from the data. A higher learning rate means larger updates to the weights and bias, while a lower learning rate means smaller updates. The choice of learning rate can significantly impact the convergence and performance of the model.
learning_rate = 0.1
#np.dot() is a function in the NumPy library that performs the dot product of two arrays. It takes two arrays as input and returns the sum of the products of corresponding elements. In the context of machine learning, it is often used to calculate the weighted sum of inputs and weights, which is a common operation in linear models such as perceptrons and neural networks.
#np.dot () multiplies each input by its corresponding weight and adds them all together
# (input1 * weight1) + (input2 * weight3) + (input3 * weight3)...
def predict (inputs, weights, bias):
    total_score = np.dot(inputs, weights) + bias
 #1 = aprove and -1 = reject
    return 1 if total_score > 0 else -1

# ==== TRAINING LOOP (learning) ====
#epoch = how many times you want the mdoel to go over the data set

epoch = 5
#using range as its not an array
for epoch in range (5):

    errors_in_epoch = 0  #to keep track of how many mistakes

    for inputs, expected_output in training_data: 

        prediction = predict(inputs, weights, bias)

        print (prediction, "<======prediction output")

        error = expected_output - prediction

        if error != 0 :
            errors_in_epoch += 1

            print ("errors in epoch ====>", errors_in_epoch)

        #update the weights and bias
            weights = weights + (learning_rate * error * inputs)

            bias = bias + (learning_rate * error)

    if errors_in_epoch == 0:
            print ("Model is done learning! : Stopping early...")
            break   

new_applicant = np.array ([0.75, 0.9, 0.85])
decision = predict (new_applicant, weights, bias)

if decision ==1:
    print ("decision_approved")
else: 
    print ("decision_denied")