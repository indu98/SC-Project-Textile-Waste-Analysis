
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from sklearn.metrics import precision_recall_fscore_support


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
dataset1 = np.loadtxt("train_1.csv", delimiter=",")
dataset2 = np.loadtxt("test_1.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset1[:,0:6]
Y = dataset1[:,6]
x_test =dataset2[:,0:6]
y_test = dataset2[:,6]

#print (X.shape)
#print (type(y_test))
# create model
model = Sequential()
model.add(Dense(21, activation='relu', input_dim=X.shape[1]))
model.add(Dense(63, activation='relu'))
model.add(Dense(1))

'''
#inspect model
print(model.input_shape)
print(model.summary())
#print (model.get_config())
'''


def forward_prop(params):
    
    n_inputs = 6
    n_hidden1 = 21 
    n_hidden2 = 63
    n_classes = 1

    # Roll-back the weights and biases
    W1 = params[0:126].reshape((n_inputs,n_hidden1))
    b1 = params[126:147].reshape((n_hidden1,))
    W2 = params[147:1470].reshape((n_hidden1,n_hidden2))
    b2 = params[1470:1533].reshape((n_hidden2,))
    W3 = params[1533:1596].reshape((n_hidden2,n_classes))
    b3 = params[1596:1597].reshape((n_classes,))
    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    a2 = np.tanh(z2)         # Logits for Layer 2
    z3 = a2.dot(W3) + b3 # Pre-activation in Layer 2
    logits = z3  

    N = len(X)
    loss=np.sum((logits[:,0]-Y)**2)
    #print(loss)
    loss=loss/N
    #print("Mean:"+str(loss))    
    return loss

def f(x):
    
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm
options = {'c1': 2.0, 'c2': 2.0, 'w':0.25}

# Call instance of PSO
dimensions = (6 * 21) + (21 * 63) + (63 * 1) + 21 + 63 + 1
optimizer = ps.single.GlobalBestPSO(n_particles=5, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, print_step=10, iters=100, verbose=3)
#print(pos)


def predict(X, pos):
    
    # Neural network architecture
    n_inputs = 6
    n_hidden1 = 21 
    n_hidden2 = 63
    n_classes = 1

    # Roll-back the weights and biases
    W1 = pos[0:126].reshape((n_inputs,n_hidden1))
    b1 = pos[126:147].reshape((n_hidden1,))
    W2 = pos[147:1470].reshape((n_hidden1,n_hidden2))
    b2 = pos[1470:1533].reshape((n_hidden2,))
    W3 = pos[1533:1596].reshape((n_hidden2,n_classes))
    b3 = pos[1596:1597].reshape((n_classes,))
    # Perform forward propagation
    z1 = X.dot(W1) + b1  # Pre-activation in Layer 1
    a1 = np.tanh(z1)     # Activation in Layer 1
    z2 = a1.dot(W2) + b2 # Pre-activation in Layer 2
    a2 = np.tanh(z2)         # Logits for Layer 2
    z3 = a2.dot(W3) + b3 # Pre-activation in Layer 2
    logits = z3 
    
    #print(logits)
    N=len(X)
    loss=np.sum((logits[:,0]-y_test)**2)
    loss=loss/N
    print("\n%s: %.6f" % ("loss", loss))

    return logits

print("Mean square Error:"+str((abs((predict(x_test, pos)) - y_test)**2).mean()))


