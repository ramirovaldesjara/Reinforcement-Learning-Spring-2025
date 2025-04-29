import math


def relu(z):
    return max(0, z)


def relu_derivative(z):
    return 1.0 if z > 0 else 0.0


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def forward_and_backward(x, W1, W2):
    # Forward
    z1_1 = W1[0][0] * x[0] + W1[0][1] * x[1]
    z1_2 = W1[1][0] * x[0] + W1[1][1] * x[1]
    h1 = relu(z1_1)
    h2 = relu(z1_2)
    z2 = W2[0] * h1 + W2[1] * h2
    out = sigmoid(z2)  # output

    # Backward
    dL_dz2 = sigmoid_derivative(z2)
    dL_dW2_0 = dL_dz2 * h1
    dL_dW2_1 = dL_dz2 * h2

    # partial L wrt z1(1) and z1(2)
    dL_dz1_1 = dL_dz2 * W2[0] * relu_derivative(z1_1)
    dL_dz1_2 = dL_dz2 * W2[1] * relu_derivative(z1_2)

    # partial L wrt W1
    dL_dW1_00 = dL_dz1_1 * x[0]
    dL_dW1_01 = dL_dz1_1 * x[1]
    dL_dW1_10 = dL_dz1_2 * x[0]
    dL_dW1_11 = dL_dz1_2 * x[1]

    return {
        'output': out,
        'dW1': [[dL_dW1_00, dL_dW1_01],
                [dL_dW1_10, dL_dW1_11]],
        'dW2': [dL_dW2_0, dL_dW2_1]
    }


x = [5, 4]
W1 = [[0.1, 0.2],
      [-0.4, 0.3]]
W2 = [0.1, 0.2]

res = forward_and_backward(x, W1, W2)
print("Output:", res['output'])
print("Grad W1:", res['dW1'])
print("Grad W2:", res['dW2'])
