from dnn_app_utils import *
from activations import *

# Loading the data (cat/non-cat)
from model import L_layer_model

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_data()

# Example of a picture
# index = 25
# plt.imshow(train_set_x_orig[index])
# plt.show()
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode(
#     "utf-8") + "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))

train_x = train_set_x_flatten / 255.
test_x = test_set_x_flatten / 255.
train_y = train_set_y
test_y = test_set_y

print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))

# 1 layer model with no hidden layer
layers_dims = [12288, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)
pred_train = predict(train_x, train_y, parameters)
pred_test = predict(test_x, test_y, parameters)
