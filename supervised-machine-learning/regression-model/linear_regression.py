import numpy as np
import matplotlib.pyplot as plt


# x_train is the input variable
# y_train is the target
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train is {x_train}")
print(f"y_train is {y_train}")


# x_train.shape returns a python tuple with an entry for each dimension
print(f"x_train.shape is {x_train.shape}")

# x_train.shape[0] is the length of the array and number of training examples
m = x_train.shape[0]  # or len(x_train)
print(f"Number of training examples is: {m}")


# training examples
for i in range(x_train.shape[0]):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")


# Plot the data points
# marker and c show the points as red crosses (the default is blue dots)
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# plt.show()


# Model Function
w = 100
b = 100

# (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,)
# (scalar) describes an argument without dimensions, just a magnitude
# np.zero(n) will return a one-dimensional numpy array with n entries
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot the model prediction
plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
