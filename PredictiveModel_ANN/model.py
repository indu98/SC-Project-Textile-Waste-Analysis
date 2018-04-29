
import numpy
from keras.models import Sequential
from keras.layers import Dense

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load dataset
dataset = numpy.loadtxt("train_1.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]

#Split into test_x and test_y variables
test_set = numpy.loadtxt("train_1.csv", delimiter=",")
x_test = test_set[2000:5000,0:6]
y_test = test_set[2000:5000,6]
print X

# create model
model = Sequential()
model.add(Dense(21, activation='relu', input_dim=6))
model.add(Dense(63, activation='relu'))
model.add(Dense(1))

#inspect model
print(model.output_shape)
print(model.summary())
print model.get_config()

# Compile model
# aplha 0.001
model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

# Fit the model
model.fit(X[0:7805,:], Y[0:7805], batch_size=32, epochs=1000, verbose=1, validation_data=(X[7805:11150,:], Y[7805:11150]))

# calculate predictions
scores = model.evaluate(x_test, y_test, batch_size=32)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict(x_test, batch_size=32)

print ("Obtained Output: ",predictions)
print ("Expected Output: ",y_test)
