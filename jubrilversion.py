import numpy as np

# Great credit(0.8), great income ($80k/yr), low debt (0.2)
app1 = np.array([0.8, 0.8, 0.2])
y1 = 1

app2 = np.array([0.3, 0.4, 0.7])
y2 = -1

app3 = np.array([0.6, 0.6, 0.8])
y3 = -1

app4 = np.array([0.9, 0.5, 0.1])
y4 = 1

training_data = [(app1, y1), (app2, y2), (app3, y3), (app4, y4)]

print(training_data)

weights = np.array([0.0, 0.0, 0.0])

bias = 0.0

learning_rate = 0.1

def predict (inputs, weights, bias):
  # np.dot multiplies each input by its corresponding weight and adds them all together
  # (Input1 * weight1) +  (Input2 * weight2).....
  total_score = np.dot(inputs, weights) + bias
   # 1 = approve and -1 = Reject
  return 1 if total_score > 0 else -1



# ==== TRAINING LOOP (LEARNING)========

epoch = 5

for epoch in range(5):

  errors_in_epoch = 0  # to track mistakes to know when to stop training

  for inputs, expected_output in training_data:

      prediction = predict(inputs, weights, bias )


      error = expected_output - prediction

      if error != 0:
          errors_in_epoch += 1

          print("errors in epoch ====>", errors_in_epoch)

          weights =  weights + (learning_rate * error * inputs)

          bias = bias + (learning_rate * error)


  if errors_in_epoch == 0:
      print("Model is done learning! Stopping early....")
      break


print("\n--- Testing on a New Applicant ---")
print(f"Inputs: Credit 0.75, Income $90k, Debt 0.85")
new_applicant = np.array([0.75, 0.9, 0.85])
decision = predict(new_applicant, weights, bias)


if decision == 1:
  print("Decision: APPROVED!")
else:
  print("Decision: Rejected!")
