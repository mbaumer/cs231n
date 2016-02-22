import numpy as np

res1 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])
res2 = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
res3 = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1], [1, 0, 0]])
res4 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
res5 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0]])
results = [res1, res2, res3, res4, res5]
# 1.03, 4.07, 0.00
# 1.01, 2.05, 2.04
# 4.08, 0.00, 1.02
# 1.00, 2.04, 2.06
# 2.03, 2.04, 1.03
final = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0]]).astype('float64')

def vote_for_best(results):
  answers = np.zeros(results[0].shape)
  predictions = np.zeros(results[0].shape)

  for idx, result in enumerate(results):
    weighting = 1+(idx/100.0)
    weighted_results = result * weighting
    answers += weighted_results
  answers = np.argmax(answers, axis=1)

  for i in range(5):   # 5 should be replaced with len(X_train[0])
    predictions[i, answers[i]] = 1.
  return predictions

predictions = vote_for_best(results)

if np.array_equal(final, predictions):
  print "You rock!"
else:
  print "Please try again :("
