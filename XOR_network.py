import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
np.random.seed(42)
input_neurons=2
hidden_neurons=2
output_neurons=1
wh=np.random.rand(input_neurons,hidden_neurons)
bh=np.random.rand(1,hidden_neurons)
wout=np.random.rand(hidden_neurons,output_neurons)
bout=np.random.rand(1,output_neurons)
lr=0.1
epochs=10000
for epoch in range(epochs):
    hidden_input=np.dot(X,wh)+bh
    hidden_output=sigmoid(hidden_input)
    final_input=np.dot(hidden_output,wout)+bout
    final_output=sigmoid(final_input)
    error=y-final_output
    d_output=error*sigmoid_derivative(final_output)
    error_hidden=d_output.dot(wout.T)
    d_hidden=error_hidden*sigmoid_derivative(hidden_output)
    wout+=hidden_output.T.dot(d_output)*lr
    bout+=np.sum(d_output,axis=0,keepdims=True)*lr
    wh+=X.T.dot(d_hidden)*lr
    bh+=np.sum(d_hidden,axis=0,keepdims=True)*lr
hidden_input = np.dot(X, wh) + bh
hidden_output = sigmoid(hidden_input)
final_output = sigmoid(np.dot(hidden_output, wout) + bout)
print("Predicted XOR Output:\n", np.round(final_output,3))
