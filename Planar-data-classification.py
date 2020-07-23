from sklearn.linear_model import LogisticRegressionCV
from dnn_app_utils import *
from model import two_layer_model

X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
plt.show()


shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]  # training set size

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' %m)

# Train the logistic regression classifier
clf = LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")


# 2 layer model
n_x = X.shape[0]
n_h = 4
n_y = Y.shape[0]
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(X, Y, layers_dims=(n_x, n_h, n_y), num_iterations=10000, print_cost=True)
plot_decision_boundary(lambda x: predict_new(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# Print accuracy
predictions = predict_new(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')


# Tuning hidden layer size
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = two_layer_model(X, Y, layers_dims=(n_x, n_h, n_y), num_iterations = 5000)
    plot_decision_boundary(lambda x: predict_new(parameters, x.T), X, Y)
    plt.show()
    predictions = predict_new(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))