import numpy as np
# from keras.models import Sequential
# from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers.core import Dense
from tensorflow.python.keras.layers import Dense

# Why XOR? Because it is a non-linearly separable problem
# XOR problem training samples


training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# XOR problem target values accordingly 
target_data = np.array([[0],[1],[1],[0]], "float32")

# we can define the neural network layers in a sequential manner
model = Sequential()
# first parameter is output dimension
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#we can define the loss function MSE or negative log lokelihood
#optimizer will find the right adjustements for the weights: SGD, Adagrad, ADAM ...
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

#epoch is an iteration over the entire dataset
#verbose 0 is silent 1 and 2 are showing results
model.fit(training_data, target_data, epochs=1000, verbose=2)

#of course we can make prediction with the trained neural network
print(model.predict(training_data).round())

# Epoch 1/1000
# 1/1 - 0s - loss: 0.2536 - binary_accuracy: 0.5000
# Epoch 2/1000
# 1/1 - 0s - loss: 0.2523 - binary_accuracy: 0.2500
# Epoch 3/1000
# 1/1 - 0s - loss: 0.2510 - binary_accuracy: 0.2500
# Epoch 4/1000
# 1/1 - 0s - loss: 0.2501 - binary_accuracy: 0.2500
# Epoch 5/1000
# 1/1 - 0s - loss: 0.2498 - binary_accuracy: 0.7500
# Epoch 6/1000
# 1/1 - 0s - loss: 0.2496 - binary_accuracy: 0.7500
# Epoch 7/1000
# 1/1 - 0s - loss: 0.2495 - binary_accuracy: 0.7500
# Epoch 8/1000
# 1/1 - 0s - loss: 0.2494 - binary_accuracy: 0.7500
# Epoch 9/1000
# 1/1 - 0s - loss: 0.2492 - binary_accuracy: 0.7500
# Epoch 10/1000
# 1/1 - 0s - loss: 0.2490 - binary_accuracy: 0.7500
# Epoch 11/1000
# 1/1 - 0s - loss: 0.2488 - binary_accuracy: 0.7500
# Epoch 12/1000
# 1/1 - 0s - loss: 0.2486 - binary_accuracy: 0.7500
# Epoch 13/1000
# 1/1 - 0s - loss: 0.2484 - binary_accuracy: 0.7500
# Epoch 14/1000
# 1/1 - 0s - loss: 0.2481 - binary_accuracy: 1.0000
# Epoch 15/1000
# 1/1 - 0s - loss: 0.2479 - binary_accuracy: 1.0000
# Epoch 16/1000
# 1/1 - 0s - loss: 0.2477 - binary_accuracy: 1.0000
# Epoch 17/1000
# 1/1 - 0s - loss: 0.2475 - binary_accuracy: 1.0000
# Epoch 18/1000
# 1/1 - 0s - loss: 0.2472 - binary_accuracy: 1.0000
# Epoch 19/1000
# 1/1 - 0s - loss: 0.2469 - binary_accuracy: 1.0000
# Epoch 20/1000
# 1/1 - 0s - loss: 0.2466 - binary_accuracy: 1.0000
# Epoch 21/1000
# 1/1 - 0s - loss: 0.2463 - binary_accuracy: 1.0000
# Epoch 22/1000
# 1/1 - 0s - loss: 0.2461 - binary_accuracy: 1.0000
# Epoch 23/1000
# 1/1 - 0s - loss: 0.2458 - binary_accuracy: 1.0000
# Epoch 24/1000
# 1/1 - 0s - loss: 0.2454 - binary_accuracy: 1.0000
# Epoch 25/1000
# 1/1 - 0s - loss: 0.2452 - binary_accuracy: 1.0000
# Epoch 26/1000
# 1/1 - 0s - loss: 0.2448 - binary_accuracy: 1.0000
# Epoch 27/1000
# 1/1 - 0s - loss: 0.2445 - binary_accuracy: 1.0000
# Epoch 28/1000
# 1/1 - 0s - loss: 0.2441 - binary_accuracy: 1.0000
# Epoch 29/1000
# 1/1 - 0s - loss: 0.2437 - binary_accuracy: 1.0000
# Epoch 30/1000
# 1/1 - 0s - loss: 0.2433 - binary_accuracy: 1.0000
# Epoch 31/1000
# 1/1 - 0s - loss: 0.2429 - binary_accuracy: 1.0000
# Epoch 32/1000
# 1/1 - 0s - loss: 0.2424 - binary_accuracy: 1.0000
# Epoch 33/1000
# 1/1 - 0s - loss: 0.2419 - binary_accuracy: 1.0000
# Epoch 34/1000
# 1/1 - 0s - loss: 0.2414 - binary_accuracy: 1.0000
# Epoch 35/1000
# 1/1 - 0s - loss: 0.2408 - binary_accuracy: 1.0000
# Epoch 36/1000
# 1/1 - 0s - loss: 0.2402 - binary_accuracy: 1.0000
# Epoch 37/1000
# 1/1 - 0s - loss: 0.2396 - binary_accuracy: 1.0000
# Epoch 38/1000
# 1/1 - 0s - loss: 0.2389 - binary_accuracy: 1.0000
# Epoch 39/1000
# 1/1 - 0s - loss: 0.2382 - binary_accuracy: 1.0000
# Epoch 40/1000
# 1/1 - 0s - loss: 0.2375 - binary_accuracy: 1.0000
# Epoch 41/1000
# 1/1 - 0s - loss: 0.2367 - binary_accuracy: 1.0000
# Epoch 42/1000
# 1/1 - 0s - loss: 0.2357 - binary_accuracy: 1.0000
# Epoch 43/1000
# 1/1 - 0s - loss: 0.2347 - binary_accuracy: 1.0000
# Epoch 44/1000
# 1/1 - 0s - loss: 0.2336 - binary_accuracy: 1.0000
# Epoch 45/1000
# 1/1 - 0s - loss: 0.2324 - binary_accuracy: 1.0000
# Epoch 46/1000
# 1/1 - 0s - loss: 0.2311 - binary_accuracy: 1.0000
# Epoch 47/1000
# 1/1 - 0s - loss: 0.2299 - binary_accuracy: 1.0000
# Epoch 48/1000
# 1/1 - 0s - loss: 0.2285 - binary_accuracy: 1.0000
# Epoch 49/1000
# 1/1 - 0s - loss: 0.2272 - binary_accuracy: 1.0000
# Epoch 50/1000
# 1/1 - 0s - loss: 0.2257 - binary_accuracy: 1.0000
# Epoch 51/1000
# 1/1 - 0s - loss: 0.2241 - binary_accuracy: 1.0000
# Epoch 52/1000
# 1/1 - 0s - loss: 0.2225 - binary_accuracy: 1.0000
# Epoch 53/1000
# 1/1 - 0s - loss: 0.2208 - binary_accuracy: 1.0000
# Epoch 54/1000
# 1/1 - 0s - loss: 0.2191 - binary_accuracy: 1.0000
# Epoch 55/1000
# 1/1 - 0s - loss: 0.2173 - binary_accuracy: 1.0000
# Epoch 56/1000
# 1/1 - 0s - loss: 0.2154 - binary_accuracy: 1.0000
# Epoch 57/1000
# 1/1 - 0s - loss: 0.2135 - binary_accuracy: 1.0000
# Epoch 58/1000
# 1/1 - 0s - loss: 0.2114 - binary_accuracy: 1.0000
# Epoch 59/1000
# 1/1 - 0s - loss: 0.2093 - binary_accuracy: 1.0000
# Epoch 60/1000
# 1/1 - 0s - loss: 0.2070 - binary_accuracy: 1.0000
# Epoch 61/1000
# 1/1 - 0s - loss: 0.2047 - binary_accuracy: 1.0000
# Epoch 62/1000
# 1/1 - 0s - loss: 0.2023 - binary_accuracy: 1.0000
# Epoch 63/1000
# 1/1 - 0s - loss: 0.1999 - binary_accuracy: 1.0000
# Epoch 64/1000
# 1/1 - 0s - loss: 0.1974 - binary_accuracy: 1.0000
# Epoch 65/1000
# 1/1 - 0s - loss: 0.1946 - binary_accuracy: 1.0000
# Epoch 66/1000
# 1/1 - 0s - loss: 0.1917 - binary_accuracy: 1.0000
# Epoch 67/1000
# 1/1 - 0s - loss: 0.1886 - binary_accuracy: 1.0000
# Epoch 68/1000
# 1/1 - 0s - loss: 0.1856 - binary_accuracy: 1.0000
# Epoch 69/1000
# 1/1 - 0s - loss: 0.1824 - binary_accuracy: 1.0000
# Epoch 70/1000
# 1/1 - 0s - loss: 0.1795 - binary_accuracy: 1.0000
# Epoch 71/1000
# 1/1 - 0s - loss: 0.1759 - binary_accuracy: 1.0000
# Epoch 72/1000
# 1/1 - 0s - loss: 0.1728 - binary_accuracy: 1.0000
# Epoch 73/1000
# 1/1 - 0s - loss: 0.1690 - binary_accuracy: 1.0000
# Epoch 74/1000
# 1/1 - 0s - loss: 0.1656 - binary_accuracy: 1.0000
# Epoch 75/1000
# 1/1 - 0s - loss: 0.1620 - binary_accuracy: 1.0000
# Epoch 76/1000
# 1/1 - 0s - loss: 0.1581 - binary_accuracy: 1.0000
# Epoch 77/1000
# 1/1 - 0s - loss: 0.1548 - binary_accuracy: 1.0000
# Epoch 78/1000
# 1/1 - 0s - loss: 0.1512 - binary_accuracy: 1.0000
# Epoch 79/1000
# 1/1 - 0s - loss: 0.1476 - binary_accuracy: 1.0000
# Epoch 80/1000
# 1/1 - 0s - loss: 0.1444 - binary_accuracy: 1.0000
# Epoch 81/1000
# 1/1 - 0s - loss: 0.1409 - binary_accuracy: 1.0000
# Epoch 82/1000
# 1/1 - 0s - loss: 0.1377 - binary_accuracy: 1.0000
# Epoch 83/1000
# 1/1 - 0s - loss: 0.1346 - binary_accuracy: 1.0000
# Epoch 84/1000
# 1/1 - 0s - loss: 0.1316 - binary_accuracy: 1.0000
# Epoch 85/1000
# 1/1 - 0s - loss: 0.1287 - binary_accuracy: 1.0000
# Epoch 86/1000
# 1/1 - 0s - loss: 0.1257 - binary_accuracy: 1.0000
# Epoch 87/1000
# 1/1 - 0s - loss: 0.1231 - binary_accuracy: 1.0000
# Epoch 88/1000
# 1/1 - 0s - loss: 0.1205 - binary_accuracy: 1.0000
# Epoch 89/1000
# 1/1 - 0s - loss: 0.1180 - binary_accuracy: 1.0000
# Epoch 90/1000
# 1/1 - 0s - loss: 0.1155 - binary_accuracy: 1.0000
# Epoch 91/1000
# 1/1 - 0s - loss: 0.1131 - binary_accuracy: 1.0000
# Epoch 92/1000
# 1/1 - 0s - loss: 0.1107 - binary_accuracy: 1.0000
# Epoch 93/1000
# 1/1 - 0s - loss: 0.1084 - binary_accuracy: 1.0000
# Epoch 94/1000
# 1/1 - 0s - loss: 0.1062 - binary_accuracy: 1.0000
# Epoch 95/1000
# 1/1 - 0s - loss: 0.1041 - binary_accuracy: 1.0000
# Epoch 96/1000
# 1/1 - 0s - loss: 0.1021 - binary_accuracy: 1.0000
# Epoch 97/1000
# 1/1 - 0s - loss: 0.1001 - binary_accuracy: 1.0000
# Epoch 98/1000
# 1/1 - 0s - loss: 0.0982 - binary_accuracy: 1.0000
# Epoch 99/1000
# 1/1 - 0s - loss: 0.0963 - binary_accuracy: 1.0000
# Epoch 100/1000
# 1/1 - 0s - loss: 0.0945 - binary_accuracy: 1.0000
# Epoch 101/1000
# 1/1 - 0s - loss: 0.0927 - binary_accuracy: 1.0000
# Epoch 102/1000
# 1/1 - 0s - loss: 0.0910 - binary_accuracy: 1.0000
# Epoch 103/1000
# 1/1 - 0s - loss: 0.0893 - binary_accuracy: 1.0000
# Epoch 104/1000
# 1/1 - 0s - loss: 0.0877 - binary_accuracy: 1.0000
# Epoch 105/1000
# 1/1 - 0s - loss: 0.0860 - binary_accuracy: 1.0000
# Epoch 106/1000
# 1/1 - 0s - loss: 0.0844 - binary_accuracy: 1.0000
# Epoch 107/1000
# 1/1 - 0s - loss: 0.0828 - binary_accuracy: 1.0000
# Epoch 108/1000
# 1/1 - 0s - loss: 0.0811 - binary_accuracy: 1.0000
# Epoch 109/1000
# 1/1 - 0s - loss: 0.0795 - binary_accuracy: 1.0000
# Epoch 110/1000
# 1/1 - 0s - loss: 0.0780 - binary_accuracy: 1.0000
# Epoch 111/1000
# 1/1 - 0s - loss: 0.0764 - binary_accuracy: 1.0000
# Epoch 112/1000
# 1/1 - 0s - loss: 0.0748 - binary_accuracy: 1.0000
# Epoch 113/1000
# 1/1 - 0s - loss: 0.0732 - binary_accuracy: 1.0000
# Epoch 114/1000
# 1/1 - 0s - loss: 0.0715 - binary_accuracy: 1.0000
# Epoch 115/1000
# 1/1 - 0s - loss: 0.0700 - binary_accuracy: 1.0000
# Epoch 116/1000
# 1/1 - 0s - loss: 0.0683 - binary_accuracy: 1.0000
# Epoch 117/1000
# 1/1 - 0s - loss: 0.0666 - binary_accuracy: 1.0000
# Epoch 118/1000
# 1/1 - 0s - loss: 0.0649 - binary_accuracy: 1.0000
# Epoch 119/1000
# 1/1 - 0s - loss: 0.0632 - binary_accuracy: 1.0000
# Epoch 120/1000
# 1/1 - 0s - loss: 0.0614 - binary_accuracy: 1.0000
# Epoch 121/1000
# 1/1 - 0s - loss: 0.0597 - binary_accuracy: 1.0000
# Epoch 122/1000
# 1/1 - 0s - loss: 0.0579 - binary_accuracy: 1.0000
# Epoch 123/1000
# 1/1 - 0s - loss: 0.0561 - binary_accuracy: 1.0000
# Epoch 124/1000
# 1/1 - 0s - loss: 0.0543 - binary_accuracy: 1.0000
# Epoch 125/1000
# 1/1 - 0s - loss: 0.0524 - binary_accuracy: 1.0000
# Epoch 126/1000
# 1/1 - 0s - loss: 0.0504 - binary_accuracy: 1.0000
# Epoch 127/1000
# 1/1 - 0s - loss: 0.0484 - binary_accuracy: 1.0000
# Epoch 128/1000
# 1/1 - 0s - loss: 0.0464 - binary_accuracy: 1.0000
# Epoch 129/1000
# 1/1 - 0s - loss: 0.0444 - binary_accuracy: 1.0000
# Epoch 130/1000
# 1/1 - 0s - loss: 0.0424 - binary_accuracy: 1.0000
# Epoch 131/1000
# 1/1 - 0s - loss: 0.0404 - binary_accuracy: 1.0000
# Epoch 132/1000
# 1/1 - 0s - loss: 0.0384 - binary_accuracy: 1.0000
# Epoch 133/1000
# 1/1 - 0s - loss: 0.0364 - binary_accuracy: 1.0000
# Epoch 134/1000
# 1/1 - 0s - loss: 0.0344 - binary_accuracy: 1.0000
# Epoch 135/1000
# 1/1 - 0s - loss: 0.0325 - binary_accuracy: 1.0000
# Epoch 136/1000
# 1/1 - 0s - loss: 0.0306 - binary_accuracy: 1.0000
# Epoch 137/1000
# 1/1 - 0s - loss: 0.0287 - binary_accuracy: 1.0000
# Epoch 138/1000
# 1/1 - 0s - loss: 0.0268 - binary_accuracy: 1.0000
# Epoch 139/1000
# 1/1 - 0s - loss: 0.0249 - binary_accuracy: 1.0000
# Epoch 140/1000
# 1/1 - 0s - loss: 0.0232 - binary_accuracy: 1.0000
# Epoch 141/1000
# 1/1 - 0s - loss: 0.0214 - binary_accuracy: 1.0000
# Epoch 142/1000
# 1/1 - 0s - loss: 0.0197 - binary_accuracy: 1.0000
# Epoch 143/1000
# 1/1 - 0s - loss: 0.0181 - binary_accuracy: 1.0000
# Epoch 144/1000
# 1/1 - 0s - loss: 0.0165 - binary_accuracy: 1.0000
# Epoch 145/1000
# 1/1 - 0s - loss: 0.0150 - binary_accuracy: 1.0000
# Epoch 146/1000
# 1/1 - 0s - loss: 0.0137 - binary_accuracy: 1.0000
# Epoch 147/1000
# 1/1 - 0s - loss: 0.0124 - binary_accuracy: 1.0000
# Epoch 148/1000
# 1/1 - 0s - loss: 0.0112 - binary_accuracy: 1.0000
# Epoch 149/1000
# 1/1 - 0s - loss: 0.0101 - binary_accuracy: 1.0000
# Epoch 150/1000
# 1/1 - 0s - loss: 0.0091 - binary_accuracy: 1.0000
# Epoch 151/1000
# 1/1 - 0s - loss: 0.0082 - binary_accuracy: 1.0000
# Epoch 152/1000
# 1/1 - 0s - loss: 0.0074 - binary_accuracy: 1.0000
# Epoch 153/1000
# 1/1 - 0s - loss: 0.0067 - binary_accuracy: 1.0000
# Epoch 154/1000
# 1/1 - 0s - loss: 0.0060 - binary_accuracy: 1.0000
# Epoch 155/1000
# 1/1 - 0s - loss: 0.0054 - binary_accuracy: 1.0000
# Epoch 156/1000
# 1/1 - 0s - loss: 0.0049 - binary_accuracy: 1.0000
# Epoch 157/1000
# 1/1 - 0s - loss: 0.0044 - binary_accuracy: 1.0000
# Epoch 158/1000
# 1/1 - 0s - loss: 0.0040 - binary_accuracy: 1.0000
# Epoch 159/1000
# 1/1 - 0s - loss: 0.0036 - binary_accuracy: 1.0000
# Epoch 160/1000
# 1/1 - 0s - loss: 0.0033 - binary_accuracy: 1.0000
# Epoch 161/1000
# 1/1 - 0s - loss: 0.0030 - binary_accuracy: 1.0000
# Epoch 162/1000
# 1/1 - 0s - loss: 0.0027 - binary_accuracy: 1.0000
# Epoch 163/1000
# 1/1 - 0s - loss: 0.0025 - binary_accuracy: 1.0000
# Epoch 164/1000
# 1/1 - 0s - loss: 0.0023 - binary_accuracy: 1.0000
# Epoch 165/1000
# 1/1 - 0s - loss: 0.0021 - binary_accuracy: 1.0000
# Epoch 166/1000
# 1/1 - 0s - loss: 0.0019 - binary_accuracy: 1.0000
# Epoch 167/1000
# 1/1 - 0s - loss: 0.0018 - binary_accuracy: 1.0000
# Epoch 168/1000
# 1/1 - 0s - loss: 0.0016 - binary_accuracy: 1.0000
# Epoch 169/1000
# 1/1 - 0s - loss: 0.0015 - binary_accuracy: 1.0000
# Epoch 170/1000
# 1/1 - 0s - loss: 0.0014 - binary_accuracy: 1.0000
# Epoch 171/1000
# 1/1 - 0s - loss: 0.0013 - binary_accuracy: 1.0000
# Epoch 172/1000
# 1/1 - 0s - loss: 0.0012 - binary_accuracy: 1.0000
# Epoch 173/1000
# 1/1 - 0s - loss: 0.0012 - binary_accuracy: 1.0000
# Epoch 174/1000
# 1/1 - 0s - loss: 0.0011 - binary_accuracy: 1.0000
# Epoch 175/1000
# 1/1 - 0s - loss: 0.0010 - binary_accuracy: 1.0000
# Epoch 176/1000
# 1/1 - 0s - loss: 9.7439e-04 - binary_accuracy: 1.0000
# Epoch 177/1000
# 1/1 - 0s - loss: 9.2231e-04 - binary_accuracy: 1.0000
# Epoch 178/1000
# 1/1 - 0s - loss: 8.7500e-04 - binary_accuracy: 1.0000
# Epoch 179/1000
# 1/1 - 0s - loss: 8.3217e-04 - binary_accuracy: 1.0000
# Epoch 180/1000
# 1/1 - 0s - loss: 7.9290e-04 - binary_accuracy: 1.0000
# Epoch 181/1000
# 1/1 - 0s - loss: 7.5676e-04 - binary_accuracy: 1.0000
# Epoch 182/1000
# 1/1 - 0s - loss: 7.2372e-04 - binary_accuracy: 1.0000
# Epoch 183/1000
# 1/1 - 0s - loss: 6.9325e-04 - binary_accuracy: 1.0000
# Epoch 184/1000
# 1/1 - 0s - loss: 6.6502e-04 - binary_accuracy: 1.0000
# Epoch 185/1000
# 1/1 - 0s - loss: 6.3889e-04 - binary_accuracy: 1.0000
# Epoch 186/1000
# 1/1 - 0s - loss: 6.1512e-04 - binary_accuracy: 1.0000
# Epoch 187/1000
# 1/1 - 0s - loss: 5.9291e-04 - binary_accuracy: 1.0000
# Epoch 188/1000
# 1/1 - 0s - loss: 5.7222e-04 - binary_accuracy: 1.0000
# Epoch 189/1000
# 1/1 - 0s - loss: 5.5271e-04 - binary_accuracy: 1.0000
# Epoch 190/1000
# 1/1 - 0s - loss: 5.3437e-04 - binary_accuracy: 1.0000
# Epoch 191/1000
# 1/1 - 0s - loss: 5.1750e-04 - binary_accuracy: 1.0000
# Epoch 192/1000
# 1/1 - 0s - loss: 5.0185e-04 - binary_accuracy: 1.0000
# Epoch 193/1000
# 1/1 - 0s - loss: 4.8710e-04 - binary_accuracy: 1.0000
# Epoch 194/1000
# 1/1 - 0s - loss: 4.7315e-04 - binary_accuracy: 1.0000
# Epoch 195/1000
# 1/1 - 0s - loss: 4.5993e-04 - binary_accuracy: 1.0000
# Epoch 196/1000
# 1/1 - 0s - loss: 4.4760e-04 - binary_accuracy: 1.0000
# Epoch 197/1000
# 1/1 - 0s - loss: 4.3602e-04 - binary_accuracy: 1.0000
# Epoch 198/1000
# 1/1 - 0s - loss: 4.2502e-04 - binary_accuracy: 1.0000
# Epoch 199/1000
# 1/1 - 0s - loss: 4.1453e-04 - binary_accuracy: 1.0000
# Epoch 200/1000
# 1/1 - 0s - loss: 4.0460e-04 - binary_accuracy: 1.0000
# Epoch 201/1000
# 1/1 - 0s - loss: 3.9508e-04 - binary_accuracy: 1.0000
# Epoch 202/1000
# 1/1 - 0s - loss: 3.8591e-04 - binary_accuracy: 1.0000
# Epoch 203/1000
# 1/1 - 0s - loss: 3.7714e-04 - binary_accuracy: 1.0000
# Epoch 204/1000
# 1/1 - 0s - loss: 3.6871e-04 - binary_accuracy: 1.0000
# Epoch 205/1000
# 1/1 - 0s - loss: 3.6071e-04 - binary_accuracy: 1.0000
# Epoch 206/1000
# 1/1 - 0s - loss: 3.5306e-04 - binary_accuracy: 1.0000
# Epoch 207/1000
# 1/1 - 0s - loss: 3.4568e-04 - binary_accuracy: 1.0000
# Epoch 208/1000
# 1/1 - 0s - loss: 3.3858e-04 - binary_accuracy: 1.0000
# Epoch 209/1000
# 1/1 - 0s - loss: 3.3172e-04 - binary_accuracy: 1.0000
# Epoch 210/1000
# 1/1 - 0s - loss: 3.2513e-04 - binary_accuracy: 1.0000
# Epoch 211/1000
# 1/1 - 0s - loss: 3.1892e-04 - binary_accuracy: 1.0000
# Epoch 212/1000
# 1/1 - 0s - loss: 3.1292e-04 - binary_accuracy: 1.0000
# Epoch 213/1000
# 1/1 - 0s - loss: 3.0704e-04 - binary_accuracy: 1.0000
# Epoch 214/1000
# 1/1 - 0s - loss: 3.0128e-04 - binary_accuracy: 1.0000
# Epoch 215/1000
# 1/1 - 0s - loss: 2.9565e-04 - binary_accuracy: 1.0000
# Epoch 216/1000
# 1/1 - 0s - loss: 2.9018e-04 - binary_accuracy: 1.0000
# Epoch 217/1000
# 1/1 - 0s - loss: 2.8493e-04 - binary_accuracy: 1.0000
# Epoch 218/1000
# 1/1 - 0s - loss: 2.7992e-04 - binary_accuracy: 1.0000
# Epoch 219/1000
# 1/1 - 0s - loss: 2.7502e-04 - binary_accuracy: 1.0000
# Epoch 220/1000
# 1/1 - 0s - loss: 2.7025e-04 - binary_accuracy: 1.0000
# Epoch 221/1000
# 1/1 - 0s - loss: 2.6564e-04 - binary_accuracy: 1.0000
# Epoch 222/1000
# 1/1 - 0s - loss: 2.6117e-04 - binary_accuracy: 1.0000
# Epoch 223/1000
# 1/1 - 0s - loss: 2.5685e-04 - binary_accuracy: 1.0000
# Epoch 224/1000
# 1/1 - 0s - loss: 2.5257e-04 - binary_accuracy: 1.0000
# Epoch 225/1000
# 1/1 - 0s - loss: 2.4843e-04 - binary_accuracy: 1.0000
# Epoch 226/1000
# 1/1 - 0s - loss: 2.4439e-04 - binary_accuracy: 1.0000
# Epoch 227/1000
# 1/1 - 0s - loss: 2.4047e-04 - binary_accuracy: 1.0000
# Epoch 228/1000
# 1/1 - 0s - loss: 2.3663e-04 - binary_accuracy: 1.0000
# Epoch 229/1000
# 1/1 - 0s - loss: 2.3289e-04 - binary_accuracy: 1.0000
# Epoch 230/1000
# 1/1 - 0s - loss: 2.2924e-04 - binary_accuracy: 1.0000
# Epoch 231/1000
# 1/1 - 0s - loss: 2.2568e-04 - binary_accuracy: 1.0000
# Epoch 232/1000
# 1/1 - 0s - loss: 2.2220e-04 - binary_accuracy: 1.0000
# Epoch 233/1000
# 1/1 - 0s - loss: 2.1880e-04 - binary_accuracy: 1.0000
# Epoch 234/1000
# 1/1 - 0s - loss: 2.1548e-04 - binary_accuracy: 1.0000
# Epoch 235/1000
# 1/1 - 0s - loss: 2.1223e-04 - binary_accuracy: 1.0000
# Epoch 236/1000
# 1/1 - 0s - loss: 2.0919e-04 - binary_accuracy: 1.0000
# Epoch 237/1000
# 1/1 - 0s - loss: 2.0646e-04 - binary_accuracy: 1.0000
# Epoch 238/1000
# 1/1 - 0s - loss: 2.0362e-04 - binary_accuracy: 1.0000
# Epoch 239/1000
# 1/1 - 0s - loss: 2.0074e-04 - binary_accuracy: 1.0000
# Epoch 240/1000
# 1/1 - 0s - loss: 1.9791e-04 - binary_accuracy: 1.0000
# Epoch 241/1000
# 1/1 - 0s - loss: 1.9517e-04 - binary_accuracy: 1.0000
# Epoch 242/1000
# 1/1 - 0s - loss: 1.9246e-04 - binary_accuracy: 1.0000
# Epoch 243/1000
# 1/1 - 0s - loss: 1.8979e-04 - binary_accuracy: 1.0000
# Epoch 244/1000
# 1/1 - 0s - loss: 1.8725e-04 - binary_accuracy: 1.0000
# Epoch 245/1000
# 1/1 - 0s - loss: 1.8481e-04 - binary_accuracy: 1.0000
# Epoch 246/1000
# 1/1 - 0s - loss: 1.8240e-04 - binary_accuracy: 1.0000
# Epoch 247/1000
# 1/1 - 0s - loss: 1.8003e-04 - binary_accuracy: 1.0000
# Epoch 248/1000
# 1/1 - 0s - loss: 1.7771e-04 - binary_accuracy: 1.0000
# Epoch 249/1000
# 1/1 - 0s - loss: 1.7542e-04 - binary_accuracy: 1.0000
# Epoch 250/1000
# 1/1 - 0s - loss: 1.7315e-04 - binary_accuracy: 1.0000
# Epoch 251/1000
# 1/1 - 0s - loss: 1.7092e-04 - binary_accuracy: 1.0000
# Epoch 252/1000
# 1/1 - 0s - loss: 1.6873e-04 - binary_accuracy: 1.0000
# Epoch 253/1000
# 1/1 - 0s - loss: 1.6659e-04 - binary_accuracy: 1.0000
# Epoch 254/1000
# 1/1 - 0s - loss: 1.6450e-04 - binary_accuracy: 1.0000
# Epoch 255/1000
# 1/1 - 0s - loss: 1.6250e-04 - binary_accuracy: 1.0000
# Epoch 256/1000
# 1/1 - 0s - loss: 1.6053e-04 - binary_accuracy: 1.0000
# Epoch 257/1000
# 1/1 - 0s - loss: 1.5853e-04 - binary_accuracy: 1.0000
# Epoch 258/1000
# 1/1 - 0s - loss: 1.5661e-04 - binary_accuracy: 1.0000
# Epoch 259/1000
# 1/1 - 0s - loss: 1.5475e-04 - binary_accuracy: 1.0000
# Epoch 260/1000
# 1/1 - 0s - loss: 1.5291e-04 - binary_accuracy: 1.0000
# Epoch 261/1000
# 1/1 - 0s - loss: 1.5109e-04 - binary_accuracy: 1.0000
# Epoch 262/1000
# 1/1 - 0s - loss: 1.4929e-04 - binary_accuracy: 1.0000
# Epoch 263/1000
# 1/1 - 0s - loss: 1.4754e-04 - binary_accuracy: 1.0000
# Epoch 264/1000
# 1/1 - 0s - loss: 1.4582e-04 - binary_accuracy: 1.0000
# Epoch 265/1000
# 1/1 - 0s - loss: 1.4412e-04 - binary_accuracy: 1.0000
# Epoch 266/1000
# 1/1 - 0s - loss: 1.4246e-04 - binary_accuracy: 1.0000
# Epoch 267/1000
# 1/1 - 0s - loss: 1.4085e-04 - binary_accuracy: 1.0000
# Epoch 268/1000
# 1/1 - 0s - loss: 1.3925e-04 - binary_accuracy: 1.0000
# Epoch 269/1000
# 1/1 - 0s - loss: 1.3768e-04 - binary_accuracy: 1.0000
# Epoch 270/1000
# 1/1 - 0s - loss: 1.3615e-04 - binary_accuracy: 1.0000
# Epoch 271/1000
# 1/1 - 0s - loss: 1.3462e-04 - binary_accuracy: 1.0000
# Epoch 272/1000
# 1/1 - 0s - loss: 1.3313e-04 - binary_accuracy: 1.0000
# Epoch 273/1000
# 1/1 - 0s - loss: 1.3167e-04 - binary_accuracy: 1.0000
# Epoch 274/1000
# 1/1 - 0s - loss: 1.3023e-04 - binary_accuracy: 1.0000
# Epoch 275/1000
# 1/1 - 0s - loss: 1.2880e-04 - binary_accuracy: 1.0000
# Epoch 276/1000
# 1/1 - 0s - loss: 1.2743e-04 - binary_accuracy: 1.0000
# Epoch 277/1000
# 1/1 - 0s - loss: 1.2604e-04 - binary_accuracy: 1.0000
# Epoch 278/1000
# 1/1 - 0s - loss: 1.2469e-04 - binary_accuracy: 1.0000
# Epoch 279/1000
# 1/1 - 0s - loss: 1.2338e-04 - binary_accuracy: 1.0000
# Epoch 280/1000
# 1/1 - 0s - loss: 1.2207e-04 - binary_accuracy: 1.0000
# Epoch 281/1000
# 1/1 - 0s - loss: 1.2077e-04 - binary_accuracy: 1.0000
# Epoch 282/1000
# 1/1 - 0s - loss: 1.1953e-04 - binary_accuracy: 1.0000
# Epoch 283/1000
# 1/1 - 0s - loss: 1.1829e-04 - binary_accuracy: 1.0000
# Epoch 284/1000
# 1/1 - 0s - loss: 1.1705e-04 - binary_accuracy: 1.0000
# Epoch 285/1000
# 1/1 - 0s - loss: 1.1585e-04 - binary_accuracy: 1.0000
# Epoch 286/1000
# 1/1 - 0s - loss: 1.1466e-04 - binary_accuracy: 1.0000
# Epoch 287/1000
# 1/1 - 0s - loss: 1.1349e-04 - binary_accuracy: 1.0000
# Epoch 288/1000
# 1/1 - 0s - loss: 1.1236e-04 - binary_accuracy: 1.0000
# Epoch 289/1000
# 1/1 - 0s - loss: 1.1123e-04 - binary_accuracy: 1.0000
# Epoch 290/1000
# 1/1 - 0s - loss: 1.1010e-04 - binary_accuracy: 1.0000
# Epoch 291/1000
# 1/1 - 0s - loss: 1.0900e-04 - binary_accuracy: 1.0000
# Epoch 292/1000
# 1/1 - 0s - loss: 1.0792e-04 - binary_accuracy: 1.0000
# Epoch 293/1000
# 1/1 - 0s - loss: 1.0687e-04 - binary_accuracy: 1.0000
# Epoch 294/1000
# 1/1 - 0s - loss: 1.0581e-04 - binary_accuracy: 1.0000
# Epoch 295/1000
# 1/1 - 0s - loss: 1.0480e-04 - binary_accuracy: 1.0000
# Epoch 296/1000
# 1/1 - 0s - loss: 1.0379e-04 - binary_accuracy: 1.0000
# Epoch 297/1000
# 1/1 - 0s - loss: 1.0281e-04 - binary_accuracy: 1.0000
# Epoch 298/1000
# 1/1 - 0s - loss: 1.0181e-04 - binary_accuracy: 1.0000
# Epoch 299/1000
# 1/1 - 0s - loss: 1.0083e-04 - binary_accuracy: 1.0000
# Epoch 300/1000
# 1/1 - 0s - loss: 9.9857e-05 - binary_accuracy: 1.0000
# Epoch 301/1000
# 1/1 - 0s - loss: 9.8923e-05 - binary_accuracy: 1.0000
# Epoch 302/1000
# 1/1 - 0s - loss: 9.7993e-05 - binary_accuracy: 1.0000
# Epoch 303/1000
# 1/1 - 0s - loss: 9.7075e-05 - binary_accuracy: 1.0000
# Epoch 304/1000
# 1/1 - 0s - loss: 9.6175e-05 - binary_accuracy: 1.0000
# Epoch 305/1000
# 1/1 - 0s - loss: 9.5286e-05 - binary_accuracy: 1.0000
# Epoch 306/1000
# 1/1 - 0s - loss: 9.4406e-05 - binary_accuracy: 1.0000
# Epoch 307/1000
# 1/1 - 0s - loss: 9.3545e-05 - binary_accuracy: 1.0000
# Epoch 308/1000
# 1/1 - 0s - loss: 9.2702e-05 - binary_accuracy: 1.0000
# Epoch 309/1000
# 1/1 - 0s - loss: 9.1861e-05 - binary_accuracy: 1.0000
# Epoch 310/1000
# 1/1 - 0s - loss: 9.1023e-05 - binary_accuracy: 1.0000
# Epoch 311/1000
# 1/1 - 0s - loss: 9.0198e-05 - binary_accuracy: 1.0000
# Epoch 312/1000
# 1/1 - 0s - loss: 8.9392e-05 - binary_accuracy: 1.0000
# Epoch 313/1000
# 1/1 - 0s - loss: 8.8597e-05 - binary_accuracy: 1.0000
# Epoch 314/1000
# 1/1 - 0s - loss: 8.7808e-05 - binary_accuracy: 1.0000
# Epoch 315/1000
# 1/1 - 0s - loss: 8.7040e-05 - binary_accuracy: 1.0000
# Epoch 316/1000
# 1/1 - 0s - loss: 8.6267e-05 - binary_accuracy: 1.0000
# Epoch 317/1000
# 1/1 - 0s - loss: 8.5525e-05 - binary_accuracy: 1.0000
# Epoch 318/1000
# 1/1 - 0s - loss: 8.4780e-05 - binary_accuracy: 1.0000
# Epoch 319/1000
# 1/1 - 0s - loss: 8.4040e-05 - binary_accuracy: 1.0000
# Epoch 320/1000
# 1/1 - 0s - loss: 8.3317e-05 - binary_accuracy: 1.0000
# Epoch 321/1000
# 1/1 - 0s - loss: 8.2598e-05 - binary_accuracy: 1.0000
# Epoch 322/1000
# 1/1 - 0s - loss: 8.1891e-05 - binary_accuracy: 1.0000
# Epoch 323/1000
# 1/1 - 0s - loss: 8.1198e-05 - binary_accuracy: 1.0000
# Epoch 324/1000
# 1/1 - 0s - loss: 8.0508e-05 - binary_accuracy: 1.0000
# Epoch 325/1000
# 1/1 - 0s - loss: 7.9837e-05 - binary_accuracy: 1.0000
# Epoch 326/1000
# 1/1 - 0s - loss: 7.9158e-05 - binary_accuracy: 1.0000
# Epoch 327/1000
# 1/1 - 0s - loss: 7.8504e-05 - binary_accuracy: 1.0000
# Epoch 328/1000
# 1/1 - 0s - loss: 7.7855e-05 - binary_accuracy: 1.0000
# Epoch 329/1000
# 1/1 - 0s - loss: 7.7211e-05 - binary_accuracy: 1.0000
# Epoch 330/1000
# 1/1 - 0s - loss: 7.6567e-05 - binary_accuracy: 1.0000
# Epoch 331/1000
# 1/1 - 0s - loss: 7.5931e-05 - binary_accuracy: 1.0000
# Epoch 332/1000
# 1/1 - 0s - loss: 7.5314e-05 - binary_accuracy: 1.0000
# Epoch 333/1000
# 1/1 - 0s - loss: 7.4707e-05 - binary_accuracy: 1.0000
# Epoch 334/1000
# 1/1 - 0s - loss: 7.4088e-05 - binary_accuracy: 1.0000
# Epoch 335/1000
# 1/1 - 0s - loss: 7.3492e-05 - binary_accuracy: 1.0000
# Epoch 336/1000
# 1/1 - 0s - loss: 7.2899e-05 - binary_accuracy: 1.0000
# Epoch 337/1000
# 1/1 - 0s - loss: 7.2327e-05 - binary_accuracy: 1.0000
# Epoch 338/1000
# 1/1 - 0s - loss: 7.1747e-05 - binary_accuracy: 1.0000
# Epoch 339/1000
# 1/1 - 0s - loss: 7.1174e-05 - binary_accuracy: 1.0000
# Epoch 340/1000
# 1/1 - 0s - loss: 7.0614e-05 - binary_accuracy: 1.0000
# Epoch 341/1000
# 1/1 - 0s - loss: 7.0060e-05 - binary_accuracy: 1.0000
# Epoch 342/1000
# 1/1 - 0s - loss: 6.9503e-05 - binary_accuracy: 1.0000
# Epoch 343/1000
# 1/1 - 0s - loss: 6.8960e-05 - binary_accuracy: 1.0000
# Epoch 344/1000
# 1/1 - 0s - loss: 6.8422e-05 - binary_accuracy: 1.0000
# Epoch 345/1000
# 1/1 - 0s - loss: 6.7893e-05 - binary_accuracy: 1.0000
# Epoch 346/1000
# 1/1 - 0s - loss: 6.7369e-05 - binary_accuracy: 1.0000
# Epoch 347/1000
# 1/1 - 0s - loss: 6.6851e-05 - binary_accuracy: 1.0000
# Epoch 348/1000
# 1/1 - 0s - loss: 6.6331e-05 - binary_accuracy: 1.0000
# Epoch 349/1000
# 1/1 - 0s - loss: 6.5829e-05 - binary_accuracy: 1.0000
# Epoch 350/1000
# 1/1 - 0s - loss: 6.5328e-05 - binary_accuracy: 1.0000
# Epoch 351/1000
# 1/1 - 0s - loss: 6.4839e-05 - binary_accuracy: 1.0000
# Epoch 352/1000
# 1/1 - 0s - loss: 6.4353e-05 - binary_accuracy: 1.0000
# Epoch 353/1000
# 1/1 - 0s - loss: 6.3868e-05 - binary_accuracy: 1.0000
# Epoch 354/1000
# 1/1 - 0s - loss: 6.3388e-05 - binary_accuracy: 1.0000
# Epoch 355/1000
# 1/1 - 0s - loss: 6.2921e-05 - binary_accuracy: 1.0000
# Epoch 356/1000
# 1/1 - 0s - loss: 6.2453e-05 - binary_accuracy: 1.0000
# Epoch 357/1000
# 1/1 - 0s - loss: 6.1989e-05 - binary_accuracy: 1.0000
# Epoch 358/1000
# 1/1 - 0s - loss: 6.1535e-05 - binary_accuracy: 1.0000
# Epoch 359/1000
# 1/1 - 0s - loss: 6.1084e-05 - binary_accuracy: 1.0000
# Epoch 360/1000
# 1/1 - 0s - loss: 6.0635e-05 - binary_accuracy: 1.0000
# Epoch 361/1000
# 1/1 - 0s - loss: 6.0192e-05 - binary_accuracy: 1.0000
# Epoch 362/1000
# 1/1 - 0s - loss: 5.9754e-05 - binary_accuracy: 1.0000
# Epoch 363/1000
# 1/1 - 0s - loss: 5.9323e-05 - binary_accuracy: 1.0000
# Epoch 364/1000
# 1/1 - 0s - loss: 5.8899e-05 - binary_accuracy: 1.0000
# Epoch 365/1000
# 1/1 - 0s - loss: 5.8474e-05 - binary_accuracy: 1.0000
# Epoch 366/1000
# 1/1 - 0s - loss: 5.8053e-05 - binary_accuracy: 1.0000
# Epoch 367/1000
# 1/1 - 0s - loss: 5.7640e-05 - binary_accuracy: 1.0000
# Epoch 368/1000
# 1/1 - 0s - loss: 5.7228e-05 - binary_accuracy: 1.0000
# Epoch 369/1000
# 1/1 - 0s - loss: 5.6830e-05 - binary_accuracy: 1.0000
# Epoch 370/1000
# 1/1 - 0s - loss: 5.6434e-05 - binary_accuracy: 1.0000
# Epoch 371/1000
# 1/1 - 0s - loss: 5.6037e-05 - binary_accuracy: 1.0000
# Epoch 372/1000
# 1/1 - 0s - loss: 5.5642e-05 - binary_accuracy: 1.0000
# Epoch 373/1000
# 1/1 - 0s - loss: 5.5251e-05 - binary_accuracy: 1.0000
# Epoch 374/1000
# 1/1 - 0s - loss: 5.4882e-05 - binary_accuracy: 1.0000
# Epoch 375/1000
# 1/1 - 0s - loss: 5.4503e-05 - binary_accuracy: 1.0000
# Epoch 376/1000
# 1/1 - 0s - loss: 5.4117e-05 - binary_accuracy: 1.0000
# Epoch 377/1000
# 1/1 - 0s - loss: 5.3748e-05 - binary_accuracy: 1.0000
# Epoch 378/1000
# 1/1 - 0s - loss: 5.3386e-05 - binary_accuracy: 1.0000
# Epoch 379/1000
# 1/1 - 0s - loss: 5.3025e-05 - binary_accuracy: 1.0000
# Epoch 380/1000
# 1/1 - 0s - loss: 5.2666e-05 - binary_accuracy: 1.0000
# Epoch 381/1000
# 1/1 - 0s - loss: 5.2307e-05 - binary_accuracy: 1.0000
# Epoch 382/1000
# 1/1 - 0s - loss: 5.1951e-05 - binary_accuracy: 1.0000
# Epoch 383/1000
# 1/1 - 0s - loss: 5.1595e-05 - binary_accuracy: 1.0000
# Epoch 384/1000
# 1/1 - 0s - loss: 5.1253e-05 - binary_accuracy: 1.0000
# Epoch 385/1000
# 1/1 - 0s - loss: 5.0915e-05 - binary_accuracy: 1.0000
# Epoch 386/1000
# 1/1 - 0s - loss: 5.0570e-05 - binary_accuracy: 1.0000
# Epoch 387/1000
# 1/1 - 0s - loss: 5.0235e-05 - binary_accuracy: 1.0000
# Epoch 388/1000
# 1/1 - 0s - loss: 4.9915e-05 - binary_accuracy: 1.0000
# Epoch 389/1000
# 1/1 - 0s - loss: 4.9596e-05 - binary_accuracy: 1.0000
# Epoch 390/1000
# 1/1 - 0s - loss: 4.9277e-05 - binary_accuracy: 1.0000
# Epoch 391/1000
# 1/1 - 0s - loss: 4.8958e-05 - binary_accuracy: 1.0000
# Epoch 392/1000
# 1/1 - 0s - loss: 4.8644e-05 - binary_accuracy: 1.0000
# Epoch 393/1000
# 1/1 - 0s - loss: 4.8331e-05 - binary_accuracy: 1.0000
# Epoch 394/1000
# 1/1 - 0s - loss: 4.8029e-05 - binary_accuracy: 1.0000
# Epoch 395/1000
# 1/1 - 0s - loss: 4.7727e-05 - binary_accuracy: 1.0000
# Epoch 396/1000
# 1/1 - 0s - loss: 4.7418e-05 - binary_accuracy: 1.0000
# Epoch 397/1000
# 1/1 - 0s - loss: 4.7123e-05 - binary_accuracy: 1.0000
# Epoch 398/1000
# 1/1 - 0s - loss: 4.6830e-05 - binary_accuracy: 1.0000
# Epoch 399/1000
# 1/1 - 0s - loss: 4.6537e-05 - binary_accuracy: 1.0000
# Epoch 400/1000
# 1/1 - 0s - loss: 4.6248e-05 - binary_accuracy: 1.0000
# Epoch 401/1000
# 1/1 - 0s - loss: 4.5960e-05 - binary_accuracy: 1.0000
# Epoch 402/1000
# 1/1 - 0s - loss: 4.5677e-05 - binary_accuracy: 1.0000
# Epoch 403/1000
# 1/1 - 0s - loss: 4.5395e-05 - binary_accuracy: 1.0000
# Epoch 404/1000
# 1/1 - 0s - loss: 4.5122e-05 - binary_accuracy: 1.0000
# Epoch 405/1000
# 1/1 - 0s - loss: 4.4845e-05 - binary_accuracy: 1.0000
# Epoch 406/1000
# 1/1 - 0s - loss: 4.4574e-05 - binary_accuracy: 1.0000
# Epoch 407/1000
# 1/1 - 0s - loss: 4.4309e-05 - binary_accuracy: 1.0000
# Epoch 408/1000
# 1/1 - 0s - loss: 4.4042e-05 - binary_accuracy: 1.0000
# Epoch 409/1000
# 1/1 - 0s - loss: 4.3776e-05 - binary_accuracy: 1.0000
# Epoch 410/1000
# 1/1 - 0s - loss: 4.3512e-05 - binary_accuracy: 1.0000
# Epoch 411/1000
# 1/1 - 0s - loss: 4.3249e-05 - binary_accuracy: 1.0000
# Epoch 412/1000
# 1/1 - 0s - loss: 4.2987e-05 - binary_accuracy: 1.0000
# Epoch 413/1000
# 1/1 - 0s - loss: 4.2734e-05 - binary_accuracy: 1.0000
# Epoch 414/1000
# 1/1 - 0s - loss: 4.2479e-05 - binary_accuracy: 1.0000
# Epoch 415/1000
# 1/1 - 0s - loss: 4.2225e-05 - binary_accuracy: 1.0000
# Epoch 416/1000
# 1/1 - 0s - loss: 4.1977e-05 - binary_accuracy: 1.0000
# Epoch 417/1000
# 1/1 - 0s - loss: 4.1729e-05 - binary_accuracy: 1.0000
# Epoch 418/1000
# 1/1 - 0s - loss: 4.1483e-05 - binary_accuracy: 1.0000
# Epoch 419/1000
# 1/1 - 0s - loss: 4.1239e-05 - binary_accuracy: 1.0000
# Epoch 420/1000
# 1/1 - 0s - loss: 4.0999e-05 - binary_accuracy: 1.0000
# Epoch 421/1000
# 1/1 - 0s - loss: 4.0762e-05 - binary_accuracy: 1.0000
# Epoch 422/1000
# 1/1 - 0s - loss: 4.0524e-05 - binary_accuracy: 1.0000
# Epoch 423/1000
# 1/1 - 0s - loss: 4.0293e-05 - binary_accuracy: 1.0000
# Epoch 424/1000
# 1/1 - 0s - loss: 4.0060e-05 - binary_accuracy: 1.0000
# Epoch 425/1000
# 1/1 - 0s - loss: 3.9830e-05 - binary_accuracy: 1.0000
# Epoch 426/1000
# 1/1 - 0s - loss: 3.9601e-05 - binary_accuracy: 1.0000
# Epoch 427/1000
# 1/1 - 0s - loss: 3.9378e-05 - binary_accuracy: 1.0000
# Epoch 428/1000
# 1/1 - 0s - loss: 3.9153e-05 - binary_accuracy: 1.0000
# Epoch 429/1000
# 1/1 - 0s - loss: 3.8934e-05 - binary_accuracy: 1.0000
# Epoch 430/1000
# 1/1 - 0s - loss: 3.8718e-05 - binary_accuracy: 1.0000
# Epoch 431/1000
# 1/1 - 0s - loss: 3.8501e-05 - binary_accuracy: 1.0000
# Epoch 432/1000
# 1/1 - 0s - loss: 3.8282e-05 - binary_accuracy: 1.0000
# Epoch 433/1000
# 1/1 - 0s - loss: 3.8066e-05 - binary_accuracy: 1.0000
# Epoch 434/1000
# 1/1 - 0s - loss: 3.7852e-05 - binary_accuracy: 1.0000
# Epoch 435/1000
# 1/1 - 0s - loss: 3.7642e-05 - binary_accuracy: 1.0000
# Epoch 436/1000
# 1/1 - 0s - loss: 3.7433e-05 - binary_accuracy: 1.0000
# Epoch 437/1000
# 1/1 - 0s - loss: 3.7224e-05 - binary_accuracy: 1.0000
# Epoch 438/1000
# 1/1 - 0s - loss: 3.7021e-05 - binary_accuracy: 1.0000
# Epoch 439/1000
# 1/1 - 0s - loss: 3.6817e-05 - binary_accuracy: 1.0000
# Epoch 440/1000
# 1/1 - 0s - loss: 3.6613e-05 - binary_accuracy: 1.0000
# Epoch 441/1000
# 1/1 - 0s - loss: 3.6410e-05 - binary_accuracy: 1.0000
# Epoch 442/1000
# 1/1 - 0s - loss: 3.6213e-05 - binary_accuracy: 1.0000
# Epoch 443/1000
# 1/1 - 0s - loss: 3.6015e-05 - binary_accuracy: 1.0000
# Epoch 444/1000
# 1/1 - 0s - loss: 3.5820e-05 - binary_accuracy: 1.0000
# Epoch 445/1000
# 1/1 - 0s - loss: 3.5627e-05 - binary_accuracy: 1.0000
# Epoch 446/1000
# 1/1 - 0s - loss: 3.5435e-05 - binary_accuracy: 1.0000
# Epoch 447/1000
# 1/1 - 0s - loss: 3.5244e-05 - binary_accuracy: 1.0000
# Epoch 448/1000
# 1/1 - 0s - loss: 3.5052e-05 - binary_accuracy: 1.0000
# Epoch 449/1000
# 1/1 - 0s - loss: 3.4861e-05 - binary_accuracy: 1.0000
# Epoch 450/1000
# 1/1 - 0s - loss: 3.4685e-05 - binary_accuracy: 1.0000
# Epoch 451/1000
# 1/1 - 0s - loss: 3.4500e-05 - binary_accuracy: 1.0000
# Epoch 452/1000
# 1/1 - 0s - loss: 3.4308e-05 - binary_accuracy: 1.0000
# Epoch 453/1000
# 1/1 - 0s - loss: 3.4130e-05 - binary_accuracy: 1.0000
# Epoch 454/1000
# 1/1 - 0s - loss: 3.3953e-05 - binary_accuracy: 1.0000
# Epoch 455/1000
# 1/1 - 0s - loss: 3.3777e-05 - binary_accuracy: 1.0000
# Epoch 456/1000
# 1/1 - 0s - loss: 3.3601e-05 - binary_accuracy: 1.0000
# Epoch 457/1000
# 1/1 - 0s - loss: 3.3425e-05 - binary_accuracy: 1.0000
# Epoch 458/1000
# 1/1 - 0s - loss: 3.3249e-05 - binary_accuracy: 1.0000
# Epoch 459/1000
# 1/1 - 0s - loss: 3.3073e-05 - binary_accuracy: 1.0000
# Epoch 460/1000
# 1/1 - 0s - loss: 3.2898e-05 - binary_accuracy: 1.0000
# Epoch 461/1000
# 1/1 - 0s - loss: 3.2724e-05 - binary_accuracy: 1.0000
# Epoch 462/1000
# 1/1 - 0s - loss: 3.2558e-05 - binary_accuracy: 1.0000
# Epoch 463/1000
# 1/1 - 0s - loss: 3.2394e-05 - binary_accuracy: 1.0000
# Epoch 464/1000
# 1/1 - 0s - loss: 3.2223e-05 - binary_accuracy: 1.0000
# Epoch 465/1000
# 1/1 - 0s - loss: 3.2054e-05 - binary_accuracy: 1.0000
# Epoch 466/1000
# 1/1 - 0s - loss: 3.1893e-05 - binary_accuracy: 1.0000
# Epoch 467/1000
# 1/1 - 0s - loss: 3.1730e-05 - binary_accuracy: 1.0000
# Epoch 468/1000
# 1/1 - 0s - loss: 3.1569e-05 - binary_accuracy: 1.0000
# Epoch 469/1000
# 1/1 - 0s - loss: 3.1408e-05 - binary_accuracy: 1.0000
# Epoch 470/1000
# 1/1 - 0s - loss: 3.1247e-05 - binary_accuracy: 1.0000
# Epoch 471/1000
# 1/1 - 0s - loss: 3.1087e-05 - binary_accuracy: 1.0000
# Epoch 472/1000
# 1/1 - 0s - loss: 3.0931e-05 - binary_accuracy: 1.0000
# Epoch 473/1000
# 1/1 - 0s - loss: 3.0775e-05 - binary_accuracy: 1.0000
# Epoch 474/1000
# 1/1 - 0s - loss: 3.0618e-05 - binary_accuracy: 1.0000
# Epoch 475/1000
# 1/1 - 0s - loss: 3.0466e-05 - binary_accuracy: 1.0000
# Epoch 476/1000
# 1/1 - 0s - loss: 3.0315e-05 - binary_accuracy: 1.0000
# Epoch 477/1000
# 1/1 - 0s - loss: 3.0162e-05 - binary_accuracy: 1.0000
# Epoch 478/1000
# 1/1 - 0s - loss: 3.0014e-05 - binary_accuracy: 1.0000
# Epoch 479/1000
# 1/1 - 0s - loss: 2.9863e-05 - binary_accuracy: 1.0000
# Epoch 480/1000
# 1/1 - 0s - loss: 2.9716e-05 - binary_accuracy: 1.0000
# Epoch 481/1000
# 1/1 - 0s - loss: 2.9570e-05 - binary_accuracy: 1.0000
# Epoch 482/1000
# 1/1 - 0s - loss: 2.9424e-05 - binary_accuracy: 1.0000
# Epoch 483/1000
# 1/1 - 0s - loss: 2.9281e-05 - binary_accuracy: 1.0000
# Epoch 484/1000
# 1/1 - 0s - loss: 2.9136e-05 - binary_accuracy: 1.0000
# Epoch 485/1000
# 1/1 - 0s - loss: 2.8994e-05 - binary_accuracy: 1.0000
# Epoch 486/1000
# 1/1 - 0s - loss: 2.8853e-05 - binary_accuracy: 1.0000
# Epoch 487/1000
# 1/1 - 0s - loss: 2.8713e-05 - binary_accuracy: 1.0000
# Epoch 488/1000
# 1/1 - 0s - loss: 2.8573e-05 - binary_accuracy: 1.0000
# Epoch 489/1000
# 1/1 - 0s - loss: 2.8437e-05 - binary_accuracy: 1.0000
# Epoch 490/1000
# 1/1 - 0s - loss: 2.8298e-05 - binary_accuracy: 1.0000
# Epoch 491/1000
# 1/1 - 0s - loss: 2.8165e-05 - binary_accuracy: 1.0000
# Epoch 492/1000
# 1/1 - 0s - loss: 2.8031e-05 - binary_accuracy: 1.0000
# Epoch 493/1000
# 1/1 - 0s - loss: 2.7898e-05 - binary_accuracy: 1.0000
# Epoch 494/1000
# 1/1 - 0s - loss: 2.7765e-05 - binary_accuracy: 1.0000
# Epoch 495/1000
# 1/1 - 0s - loss: 2.7632e-05 - binary_accuracy: 1.0000
# Epoch 496/1000
# 1/1 - 0s - loss: 2.7499e-05 - binary_accuracy: 1.0000
# Epoch 497/1000
# 1/1 - 0s - loss: 2.7366e-05 - binary_accuracy: 1.0000
# Epoch 498/1000
# 1/1 - 0s - loss: 2.7237e-05 - binary_accuracy: 1.0000
# Epoch 499/1000
# 1/1 - 0s - loss: 2.7110e-05 - binary_accuracy: 1.0000
# Epoch 500/1000
# 1/1 - 0s - loss: 2.6978e-05 - binary_accuracy: 1.0000
# Epoch 501/1000
# 1/1 - 0s - loss: 2.6852e-05 - binary_accuracy: 1.0000
# Epoch 502/1000
# 1/1 - 0s - loss: 2.6726e-05 - binary_accuracy: 1.0000
# Epoch 503/1000
# 1/1 - 0s - loss: 2.6600e-05 - binary_accuracy: 1.0000
# Epoch 504/1000
# 1/1 - 0s - loss: 2.6477e-05 - binary_accuracy: 1.0000
# Epoch 505/1000
# 1/1 - 0s - loss: 2.6352e-05 - binary_accuracy: 1.0000
# Epoch 506/1000
# 1/1 - 0s - loss: 2.6231e-05 - binary_accuracy: 1.0000
# Epoch 507/1000
# 1/1 - 0s - loss: 2.6112e-05 - binary_accuracy: 1.0000
# Epoch 508/1000
# 1/1 - 0s - loss: 2.5992e-05 - binary_accuracy: 1.0000
# Epoch 509/1000
# 1/1 - 0s - loss: 2.5871e-05 - binary_accuracy: 1.0000
# Epoch 510/1000
# 1/1 - 0s - loss: 2.5752e-05 - binary_accuracy: 1.0000
# Epoch 511/1000
# 1/1 - 0s - loss: 2.5632e-05 - binary_accuracy: 1.0000
# Epoch 512/1000
# 1/1 - 0s - loss: 2.5512e-05 - binary_accuracy: 1.0000
# Epoch 513/1000
# 1/1 - 0s - loss: 2.5401e-05 - binary_accuracy: 1.0000
# Epoch 514/1000
# 1/1 - 0s - loss: 2.5286e-05 - binary_accuracy: 1.0000
# Epoch 515/1000
# 1/1 - 0s - loss: 2.5166e-05 - binary_accuracy: 1.0000
# Epoch 516/1000
# 1/1 - 0s - loss: 2.5051e-05 - binary_accuracy: 1.0000
# Epoch 517/1000
# 1/1 - 0s - loss: 2.4941e-05 - binary_accuracy: 1.0000
# Epoch 518/1000
# 1/1 - 0s - loss: 2.4829e-05 - binary_accuracy: 1.0000
# Epoch 519/1000
# 1/1 - 0s - loss: 2.4717e-05 - binary_accuracy: 1.0000
# Epoch 520/1000
# 1/1 - 0s - loss: 2.4606e-05 - binary_accuracy: 1.0000
# Epoch 521/1000
# 1/1 - 0s - loss: 2.4495e-05 - binary_accuracy: 1.0000
# Epoch 522/1000
# 1/1 - 0s - loss: 2.4384e-05 - binary_accuracy: 1.0000
# Epoch 523/1000
# 1/1 - 0s - loss: 2.4273e-05 - binary_accuracy: 1.0000
# Epoch 524/1000
# 1/1 - 0s - loss: 2.4164e-05 - binary_accuracy: 1.0000
# Epoch 525/1000
# 1/1 - 0s - loss: 2.4057e-05 - binary_accuracy: 1.0000
# Epoch 526/1000
# 1/1 - 0s - loss: 2.3948e-05 - binary_accuracy: 1.0000
# Epoch 527/1000
# 1/1 - 0s - loss: 2.3843e-05 - binary_accuracy: 1.0000
# Epoch 528/1000
# 1/1 - 0s - loss: 2.3737e-05 - binary_accuracy: 1.0000
# Epoch 529/1000
# 1/1 - 0s - loss: 2.3632e-05 - binary_accuracy: 1.0000
# Epoch 530/1000
# 1/1 - 0s - loss: 2.3531e-05 - binary_accuracy: 1.0000
# Epoch 531/1000
# 1/1 - 0s - loss: 2.3426e-05 - binary_accuracy: 1.0000
# Epoch 532/1000
# 1/1 - 0s - loss: 2.3323e-05 - binary_accuracy: 1.0000
# Epoch 533/1000
# 1/1 - 0s - loss: 2.3222e-05 - binary_accuracy: 1.0000
# Epoch 534/1000
# 1/1 - 0s - loss: 2.3121e-05 - binary_accuracy: 1.0000
# Epoch 535/1000
# 1/1 - 0s - loss: 2.3021e-05 - binary_accuracy: 1.0000
# Epoch 536/1000
# 1/1 - 0s - loss: 2.2920e-05 - binary_accuracy: 1.0000
# Epoch 537/1000
# 1/1 - 0s - loss: 2.2819e-05 - binary_accuracy: 1.0000
# Epoch 538/1000
# 1/1 - 0s - loss: 2.2720e-05 - binary_accuracy: 1.0000
# Epoch 539/1000
# 1/1 - 0s - loss: 2.2621e-05 - binary_accuracy: 1.0000
# Epoch 540/1000
# 1/1 - 0s - loss: 2.2524e-05 - binary_accuracy: 1.0000
# Epoch 541/1000
# 1/1 - 0s - loss: 2.2429e-05 - binary_accuracy: 1.0000
# Epoch 542/1000
# 1/1 - 0s - loss: 2.2333e-05 - binary_accuracy: 1.0000
# Epoch 543/1000
# 1/1 - 0s - loss: 2.2237e-05 - binary_accuracy: 1.0000
# Epoch 544/1000
# 1/1 - 0s - loss: 2.2141e-05 - binary_accuracy: 1.0000
# Epoch 545/1000
# 1/1 - 0s - loss: 2.2045e-05 - binary_accuracy: 1.0000
# Epoch 546/1000
# 1/1 - 0s - loss: 2.1954e-05 - binary_accuracy: 1.0000
# Epoch 547/1000
# 1/1 - 0s - loss: 2.1860e-05 - binary_accuracy: 1.0000
# Epoch 548/1000
# 1/1 - 0s - loss: 2.1765e-05 - binary_accuracy: 1.0000
# Epoch 549/1000
# 1/1 - 0s - loss: 2.1673e-05 - binary_accuracy: 1.0000
# Epoch 550/1000
# 1/1 - 0s - loss: 2.1583e-05 - binary_accuracy: 1.0000
# Epoch 551/1000
# 1/1 - 0s - loss: 2.1491e-05 - binary_accuracy: 1.0000
# Epoch 552/1000
# 1/1 - 0s - loss: 2.1400e-05 - binary_accuracy: 1.0000
# Epoch 553/1000
# 1/1 - 0s - loss: 2.1311e-05 - binary_accuracy: 1.0000
# Epoch 554/1000
# 1/1 - 0s - loss: 2.1222e-05 - binary_accuracy: 1.0000
# Epoch 555/1000
# 1/1 - 0s - loss: 2.1133e-05 - binary_accuracy: 1.0000
# Epoch 556/1000
# 1/1 - 0s - loss: 2.1047e-05 - binary_accuracy: 1.0000
# Epoch 557/1000
# 1/1 - 0s - loss: 2.0958e-05 - binary_accuracy: 1.0000
# Epoch 558/1000
# 1/1 - 0s - loss: 2.0871e-05 - binary_accuracy: 1.0000
# Epoch 559/1000
# 1/1 - 0s - loss: 2.0786e-05 - binary_accuracy: 1.0000
# Epoch 560/1000
# 1/1 - 0s - loss: 2.0701e-05 - binary_accuracy: 1.0000
# Epoch 561/1000
# 1/1 - 0s - loss: 2.0615e-05 - binary_accuracy: 1.0000
# Epoch 562/1000
# 1/1 - 0s - loss: 2.0530e-05 - binary_accuracy: 1.0000
# Epoch 563/1000
# 1/1 - 0s - loss: 2.0444e-05 - binary_accuracy: 1.0000
# Epoch 564/1000
# 1/1 - 0s - loss: 2.0359e-05 - binary_accuracy: 1.0000
# Epoch 565/1000
# 1/1 - 0s - loss: 2.0279e-05 - binary_accuracy: 1.0000
# Epoch 566/1000
# 1/1 - 0s - loss: 2.0197e-05 - binary_accuracy: 1.0000
# Epoch 567/1000
# 1/1 - 0s - loss: 2.0110e-05 - binary_accuracy: 1.0000
# Epoch 568/1000
# 1/1 - 0s - loss: 2.0030e-05 - binary_accuracy: 1.0000
# Epoch 569/1000
# 1/1 - 0s - loss: 1.9948e-05 - binary_accuracy: 1.0000
# Epoch 570/1000
# 1/1 - 0s - loss: 1.9868e-05 - binary_accuracy: 1.0000
# Epoch 571/1000
# 1/1 - 0s - loss: 1.9786e-05 - binary_accuracy: 1.0000
# Epoch 572/1000
# 1/1 - 0s - loss: 1.9705e-05 - binary_accuracy: 1.0000
# Epoch 573/1000
# 1/1 - 0s - loss: 1.9623e-05 - binary_accuracy: 1.0000
# Epoch 574/1000
# 1/1 - 0s - loss: 1.9542e-05 - binary_accuracy: 1.0000
# Epoch 575/1000
# 1/1 - 0s - loss: 1.9460e-05 - binary_accuracy: 1.0000
# Epoch 576/1000
# 1/1 - 0s - loss: 1.9378e-05 - binary_accuracy: 1.0000
# Epoch 577/1000
# 1/1 - 0s - loss: 1.9303e-05 - binary_accuracy: 1.0000
# Epoch 578/1000
# 1/1 - 0s - loss: 1.9224e-05 - binary_accuracy: 1.0000
# Epoch 579/1000
# 1/1 - 0s - loss: 1.9143e-05 - binary_accuracy: 1.0000
# Epoch 580/1000
# 1/1 - 0s - loss: 1.9064e-05 - binary_accuracy: 1.0000
# Epoch 581/1000
# 1/1 - 0s - loss: 1.8987e-05 - binary_accuracy: 1.0000
# Epoch 582/1000
# 1/1 - 0s - loss: 1.8911e-05 - binary_accuracy: 1.0000
# Epoch 583/1000
# 1/1 - 0s - loss: 1.8834e-05 - binary_accuracy: 1.0000
# Epoch 584/1000
# 1/1 - 0s - loss: 1.8756e-05 - binary_accuracy: 1.0000
# Epoch 585/1000
# 1/1 - 0s - loss: 1.8680e-05 - binary_accuracy: 1.0000
# Epoch 586/1000
# 1/1 - 0s - loss: 1.8603e-05 - binary_accuracy: 1.0000
# Epoch 587/1000
# 1/1 - 0s - loss: 1.8526e-05 - binary_accuracy: 1.0000
# Epoch 588/1000
# 1/1 - 0s - loss: 1.8456e-05 - binary_accuracy: 1.0000
# Epoch 589/1000
# 1/1 - 0s - loss: 1.8382e-05 - binary_accuracy: 1.0000
# Epoch 590/1000
# 1/1 - 0s - loss: 1.8305e-05 - binary_accuracy: 1.0000
# Epoch 591/1000
# 1/1 - 0s - loss: 1.8230e-05 - binary_accuracy: 1.0000
# Epoch 592/1000
# 1/1 - 0s - loss: 1.8159e-05 - binary_accuracy: 1.0000
# Epoch 593/1000
# 1/1 - 0s - loss: 1.8087e-05 - binary_accuracy: 1.0000
# Epoch 594/1000
# 1/1 - 0s - loss: 1.8015e-05 - binary_accuracy: 1.0000
# Epoch 595/1000
# 1/1 - 0s - loss: 1.7943e-05 - binary_accuracy: 1.0000
# Epoch 596/1000
# 1/1 - 0s - loss: 1.7871e-05 - binary_accuracy: 1.0000
# Epoch 597/1000
# 1/1 - 0s - loss: 1.7799e-05 - binary_accuracy: 1.0000
# Epoch 598/1000
# 1/1 - 0s - loss: 1.7727e-05 - binary_accuracy: 1.0000
# Epoch 599/1000
# 1/1 - 0s - loss: 1.7655e-05 - binary_accuracy: 1.0000
# Epoch 600/1000
# 1/1 - 0s - loss: 1.7585e-05 - binary_accuracy: 1.0000
# Epoch 601/1000
# 1/1 - 0s - loss: 1.7517e-05 - binary_accuracy: 1.0000
# Epoch 602/1000
# 1/1 - 0s - loss: 1.7448e-05 - binary_accuracy: 1.0000
# Epoch 603/1000
# 1/1 - 0s - loss: 1.7380e-05 - binary_accuracy: 1.0000
# Epoch 604/1000
# 1/1 - 0s - loss: 1.7311e-05 - binary_accuracy: 1.0000
# Epoch 605/1000
# 1/1 - 0s - loss: 1.7242e-05 - binary_accuracy: 1.0000
# Epoch 606/1000
# 1/1 - 0s - loss: 1.7177e-05 - binary_accuracy: 1.0000
# Epoch 607/1000
# 1/1 - 0s - loss: 1.7110e-05 - binary_accuracy: 1.0000
# Epoch 608/1000
# 1/1 - 0s - loss: 1.7041e-05 - binary_accuracy: 1.0000
# Epoch 609/1000
# 1/1 - 0s - loss: 1.6976e-05 - binary_accuracy: 1.0000
# Epoch 610/1000
# 1/1 - 0s - loss: 1.6910e-05 - binary_accuracy: 1.0000
# Epoch 611/1000
# 1/1 - 0s - loss: 1.6844e-05 - binary_accuracy: 1.0000
# Epoch 612/1000
# 1/1 - 0s - loss: 1.6779e-05 - binary_accuracy: 1.0000
# Epoch 613/1000
# 1/1 - 0s - loss: 1.6713e-05 - binary_accuracy: 1.0000
# Epoch 614/1000
# 1/1 - 0s - loss: 1.6652e-05 - binary_accuracy: 1.0000
# Epoch 615/1000
# 1/1 - 0s - loss: 1.6588e-05 - binary_accuracy: 1.0000
# Epoch 616/1000
# 1/1 - 0s - loss: 1.6522e-05 - binary_accuracy: 1.0000
# Epoch 617/1000
# 1/1 - 0s - loss: 1.6459e-05 - binary_accuracy: 1.0000
# Epoch 618/1000
# 1/1 - 0s - loss: 1.6397e-05 - binary_accuracy: 1.0000
# Epoch 619/1000
# 1/1 - 0s - loss: 1.6334e-05 - binary_accuracy: 1.0000
# Epoch 620/1000
# 1/1 - 0s - loss: 1.6271e-05 - binary_accuracy: 1.0000
# Epoch 621/1000
# 1/1 - 0s - loss: 1.6210e-05 - binary_accuracy: 1.0000
# Epoch 622/1000
# 1/1 - 0s - loss: 1.6148e-05 - binary_accuracy: 1.0000
# Epoch 623/1000
# 1/1 - 0s - loss: 1.6087e-05 - binary_accuracy: 1.0000
# Epoch 624/1000
# 1/1 - 0s - loss: 1.6027e-05 - binary_accuracy: 1.0000
# Epoch 625/1000
# 1/1 - 0s - loss: 1.5967e-05 - binary_accuracy: 1.0000
# Epoch 626/1000
# 1/1 - 0s - loss: 1.5907e-05 - binary_accuracy: 1.0000
# Epoch 627/1000
# 1/1 - 0s - loss: 1.5847e-05 - binary_accuracy: 1.0000
# Epoch 628/1000
# 1/1 - 0s - loss: 1.5786e-05 - binary_accuracy: 1.0000
# Epoch 629/1000
# 1/1 - 0s - loss: 1.5726e-05 - binary_accuracy: 1.0000
# Epoch 630/1000
# 1/1 - 0s - loss: 1.5668e-05 - binary_accuracy: 1.0000
# Epoch 631/1000
# 1/1 - 0s - loss: 1.5610e-05 - binary_accuracy: 1.0000
# Epoch 632/1000
# 1/1 - 0s - loss: 1.5552e-05 - binary_accuracy: 1.0000
# Epoch 633/1000
# 1/1 - 0s - loss: 1.5494e-05 - binary_accuracy: 1.0000
# Epoch 634/1000
# 1/1 - 0s - loss: 1.5437e-05 - binary_accuracy: 1.0000
# Epoch 635/1000
# 1/1 - 0s - loss: 1.5379e-05 - binary_accuracy: 1.0000
# Epoch 636/1000
# 1/1 - 0s - loss: 1.5322e-05 - binary_accuracy: 1.0000
# Epoch 637/1000
# 1/1 - 0s - loss: 1.5265e-05 - binary_accuracy: 1.0000
# Epoch 638/1000
# 1/1 - 0s - loss: 1.5209e-05 - binary_accuracy: 1.0000
# Epoch 639/1000
# 1/1 - 0s - loss: 1.5153e-05 - binary_accuracy: 1.0000
# Epoch 640/1000
# 1/1 - 0s - loss: 1.5098e-05 - binary_accuracy: 1.0000
# Epoch 641/1000
# 1/1 - 0s - loss: 1.5043e-05 - binary_accuracy: 1.0000
# Epoch 642/1000
# 1/1 - 0s - loss: 1.4988e-05 - binary_accuracy: 1.0000
# Epoch 643/1000
# 1/1 - 0s - loss: 1.4933e-05 - binary_accuracy: 1.0000
# Epoch 644/1000
# 1/1 - 0s - loss: 1.4879e-05 - binary_accuracy: 1.0000
# Epoch 645/1000
# 1/1 - 0s - loss: 1.4824e-05 - binary_accuracy: 1.0000
# Epoch 646/1000
# 1/1 - 0s - loss: 1.4771e-05 - binary_accuracy: 1.0000
# Epoch 647/1000
# 1/1 - 0s - loss: 1.4718e-05 - binary_accuracy: 1.0000
# Epoch 648/1000
# 1/1 - 0s - loss: 1.4666e-05 - binary_accuracy: 1.0000
# Epoch 649/1000
# 1/1 - 0s - loss: 1.4613e-05 - binary_accuracy: 1.0000
# Epoch 650/1000
# 1/1 - 0s - loss: 1.4560e-05 - binary_accuracy: 1.0000
# Epoch 651/1000
# 1/1 - 0s - loss: 1.4507e-05 - binary_accuracy: 1.0000
# Epoch 652/1000
# 1/1 - 0s - loss: 1.4454e-05 - binary_accuracy: 1.0000
# Epoch 653/1000
# 1/1 - 0s - loss: 1.4403e-05 - binary_accuracy: 1.0000
# Epoch 654/1000
# 1/1 - 0s - loss: 1.4351e-05 - binary_accuracy: 1.0000
# Epoch 655/1000
# 1/1 - 0s - loss: 1.4299e-05 - binary_accuracy: 1.0000
# Epoch 656/1000
# 1/1 - 0s - loss: 1.4248e-05 - binary_accuracy: 1.0000
# Epoch 657/1000
# 1/1 - 0s - loss: 1.4198e-05 - binary_accuracy: 1.0000
# Epoch 658/1000
# 1/1 - 0s - loss: 1.4147e-05 - binary_accuracy: 1.0000
# Epoch 659/1000
# 1/1 - 0s - loss: 1.4097e-05 - binary_accuracy: 1.0000
# Epoch 660/1000
# 1/1 - 0s - loss: 1.4049e-05 - binary_accuracy: 1.0000
# Epoch 661/1000
# 1/1 - 0s - loss: 1.3998e-05 - binary_accuracy: 1.0000
# Epoch 662/1000
# 1/1 - 0s - loss: 1.3948e-05 - binary_accuracy: 1.0000
# Epoch 663/1000
# 1/1 - 0s - loss: 1.3899e-05 - binary_accuracy: 1.0000
# Epoch 664/1000
# 1/1 - 0s - loss: 1.3851e-05 - binary_accuracy: 1.0000
# Epoch 665/1000
# 1/1 - 0s - loss: 1.3803e-05 - binary_accuracy: 1.0000
# Epoch 666/1000
# 1/1 - 0s - loss: 1.3754e-05 - binary_accuracy: 1.0000
# Epoch 667/1000
# 1/1 - 0s - loss: 1.3705e-05 - binary_accuracy: 1.0000
# Epoch 668/1000
# 1/1 - 0s - loss: 1.3657e-05 - binary_accuracy: 1.0000
# Epoch 669/1000
# 1/1 - 0s - loss: 1.3610e-05 - binary_accuracy: 1.0000
# Epoch 670/1000
# 1/1 - 0s - loss: 1.3563e-05 - binary_accuracy: 1.0000
# Epoch 671/1000
# 1/1 - 0s - loss: 1.3516e-05 - binary_accuracy: 1.0000
# Epoch 672/1000
# 1/1 - 0s - loss: 1.3470e-05 - binary_accuracy: 1.0000
# Epoch 673/1000
# 1/1 - 0s - loss: 1.3423e-05 - binary_accuracy: 1.0000
# Epoch 674/1000
# 1/1 - 0s - loss: 1.3376e-05 - binary_accuracy: 1.0000
# Epoch 675/1000
# 1/1 - 0s - loss: 1.3330e-05 - binary_accuracy: 1.0000
# Epoch 676/1000
# 1/1 - 0s - loss: 1.3284e-05 - binary_accuracy: 1.0000
# Epoch 677/1000
# 1/1 - 0s - loss: 1.3238e-05 - binary_accuracy: 1.0000
# Epoch 678/1000
# 1/1 - 0s - loss: 1.3193e-05 - binary_accuracy: 1.0000
# Epoch 679/1000
# 1/1 - 0s - loss: 1.3148e-05 - binary_accuracy: 1.0000
# Epoch 680/1000
# 1/1 - 0s - loss: 1.3103e-05 - binary_accuracy: 1.0000
# Epoch 681/1000
# 1/1 - 0s - loss: 1.3059e-05 - binary_accuracy: 1.0000
# Epoch 682/1000
# 1/1 - 0s - loss: 1.3014e-05 - binary_accuracy: 1.0000
# Epoch 683/1000
# 1/1 - 0s - loss: 1.2969e-05 - binary_accuracy: 1.0000
# Epoch 684/1000
# 1/1 - 0s - loss: 1.2926e-05 - binary_accuracy: 1.0000
# Epoch 685/1000
# 1/1 - 0s - loss: 1.2882e-05 - binary_accuracy: 1.0000
# Epoch 686/1000
# 1/1 - 0s - loss: 1.2837e-05 - binary_accuracy: 1.0000
# Epoch 687/1000
# 1/1 - 0s - loss: 1.2795e-05 - binary_accuracy: 1.0000
# Epoch 688/1000
# 1/1 - 0s - loss: 1.2752e-05 - binary_accuracy: 1.0000
# Epoch 689/1000
# 1/1 - 0s - loss: 1.2709e-05 - binary_accuracy: 1.0000
# Epoch 690/1000
# 1/1 - 0s - loss: 1.2666e-05 - binary_accuracy: 1.0000
# Epoch 691/1000
# 1/1 - 0s - loss: 1.2623e-05 - binary_accuracy: 1.0000
# Epoch 692/1000
# 1/1 - 0s - loss: 1.2582e-05 - binary_accuracy: 1.0000
# Epoch 693/1000
# 1/1 - 0s - loss: 1.2540e-05 - binary_accuracy: 1.0000
# Epoch 694/1000
# 1/1 - 0s - loss: 1.2497e-05 - binary_accuracy: 1.0000
# Epoch 695/1000
# 1/1 - 0s - loss: 1.2456e-05 - binary_accuracy: 1.0000
# Epoch 696/1000
# 1/1 - 0s - loss: 1.2414e-05 - binary_accuracy: 1.0000
# Epoch 697/1000
# 1/1 - 0s - loss: 1.2373e-05 - binary_accuracy: 1.0000
# Epoch 698/1000
# 1/1 - 0s - loss: 1.2332e-05 - binary_accuracy: 1.0000
# Epoch 699/1000
# 1/1 - 0s - loss: 1.2290e-05 - binary_accuracy: 1.0000
# Epoch 700/1000
# 1/1 - 0s - loss: 1.2249e-05 - binary_accuracy: 1.0000
# Epoch 701/1000
# 1/1 - 0s - loss: 1.2209e-05 - binary_accuracy: 1.0000
# Epoch 702/1000
# 1/1 - 0s - loss: 1.2169e-05 - binary_accuracy: 1.0000
# Epoch 703/1000
# 1/1 - 0s - loss: 1.2129e-05 - binary_accuracy: 1.0000
# Epoch 704/1000
# 1/1 - 0s - loss: 1.2090e-05 - binary_accuracy: 1.0000
# Epoch 705/1000
# 1/1 - 0s - loss: 1.2050e-05 - binary_accuracy: 1.0000
# Epoch 706/1000
# 1/1 - 0s - loss: 1.2010e-05 - binary_accuracy: 1.0000
# Epoch 707/1000
# 1/1 - 0s - loss: 1.1972e-05 - binary_accuracy: 1.0000
# Epoch 708/1000
# 1/1 - 0s - loss: 1.1932e-05 - binary_accuracy: 1.0000
# Epoch 709/1000
# 1/1 - 0s - loss: 1.1894e-05 - binary_accuracy: 1.0000
# Epoch 710/1000
# 1/1 - 0s - loss: 1.1856e-05 - binary_accuracy: 1.0000
# Epoch 711/1000
# 1/1 - 0s - loss: 1.1817e-05 - binary_accuracy: 1.0000
# Epoch 712/1000
# 1/1 - 0s - loss: 1.1779e-05 - binary_accuracy: 1.0000
# Epoch 713/1000
# 1/1 - 0s - loss: 1.1741e-05 - binary_accuracy: 1.0000
# Epoch 714/1000
# 1/1 - 0s - loss: 1.1702e-05 - binary_accuracy: 1.0000
# Epoch 715/1000
# 1/1 - 0s - loss: 1.1664e-05 - binary_accuracy: 1.0000
# Epoch 716/1000
# 1/1 - 0s - loss: 1.1626e-05 - binary_accuracy: 1.0000
# Epoch 717/1000
# 1/1 - 0s - loss: 1.1589e-05 - binary_accuracy: 1.0000
# Epoch 718/1000
# 1/1 - 0s - loss: 1.1552e-05 - binary_accuracy: 1.0000
# Epoch 719/1000
# 1/1 - 0s - loss: 1.1515e-05 - binary_accuracy: 1.0000
# Epoch 720/1000
# 1/1 - 0s - loss: 1.1478e-05 - binary_accuracy: 1.0000
# Epoch 721/1000
# 1/1 - 0s - loss: 1.1441e-05 - binary_accuracy: 1.0000
# Epoch 722/1000
# 1/1 - 0s - loss: 1.1405e-05 - binary_accuracy: 1.0000
# Epoch 723/1000
# 1/1 - 0s - loss: 1.1368e-05 - binary_accuracy: 1.0000
# Epoch 724/1000
# 1/1 - 0s - loss: 1.1333e-05 - binary_accuracy: 1.0000
# Epoch 725/1000
# 1/1 - 0s - loss: 1.1297e-05 - binary_accuracy: 1.0000
# Epoch 726/1000
# 1/1 - 0s - loss: 1.1260e-05 - binary_accuracy: 1.0000
# Epoch 727/1000
# 1/1 - 0s - loss: 1.1225e-05 - binary_accuracy: 1.0000
# Epoch 728/1000
# 1/1 - 0s - loss: 1.1189e-05 - binary_accuracy: 1.0000
# Epoch 729/1000
# 1/1 - 0s - loss: 1.1154e-05 - binary_accuracy: 1.0000
# Epoch 730/1000
# 1/1 - 0s - loss: 1.1118e-05 - binary_accuracy: 1.0000
# Epoch 731/1000
# 1/1 - 0s - loss: 1.1084e-05 - binary_accuracy: 1.0000
# Epoch 732/1000
# 1/1 - 0s - loss: 1.1049e-05 - binary_accuracy: 1.0000
# Epoch 733/1000
# 1/1 - 0s - loss: 1.1015e-05 - binary_accuracy: 1.0000
# Epoch 734/1000
# 1/1 - 0s - loss: 1.0981e-05 - binary_accuracy: 1.0000
# Epoch 735/1000
# 1/1 - 0s - loss: 1.0947e-05 - binary_accuracy: 1.0000
# Epoch 736/1000
# 1/1 - 0s - loss: 1.0912e-05 - binary_accuracy: 1.0000
# Epoch 737/1000
# 1/1 - 0s - loss: 1.0878e-05 - binary_accuracy: 1.0000
# Epoch 738/1000
# 1/1 - 0s - loss: 1.0844e-05 - binary_accuracy: 1.0000
# Epoch 739/1000
# 1/1 - 0s - loss: 1.0810e-05 - binary_accuracy: 1.0000
# Epoch 740/1000
# 1/1 - 0s - loss: 1.0776e-05 - binary_accuracy: 1.0000
# Epoch 741/1000
# 1/1 - 0s - loss: 1.0742e-05 - binary_accuracy: 1.0000
# Epoch 742/1000
# 1/1 - 0s - loss: 1.0709e-05 - binary_accuracy: 1.0000
# Epoch 743/1000
# 1/1 - 0s - loss: 1.0676e-05 - binary_accuracy: 1.0000
# Epoch 744/1000
# 1/1 - 0s - loss: 1.0644e-05 - binary_accuracy: 1.0000
# Epoch 745/1000
# 1/1 - 0s - loss: 1.0611e-05 - binary_accuracy: 1.0000
# Epoch 746/1000
# 1/1 - 0s - loss: 1.0578e-05 - binary_accuracy: 1.0000
# Epoch 747/1000
# 1/1 - 0s - loss: 1.0545e-05 - binary_accuracy: 1.0000
# Epoch 748/1000
# 1/1 - 0s - loss: 1.0514e-05 - binary_accuracy: 1.0000
# Epoch 749/1000
# 1/1 - 0s - loss: 1.0481e-05 - binary_accuracy: 1.0000
# Epoch 750/1000
# 1/1 - 0s - loss: 1.0448e-05 - binary_accuracy: 1.0000
# Epoch 751/1000
# 1/1 - 0s - loss: 1.0417e-05 - binary_accuracy: 1.0000
# Epoch 752/1000
# 1/1 - 0s - loss: 1.0385e-05 - binary_accuracy: 1.0000
# Epoch 753/1000
# 1/1 - 0s - loss: 1.0354e-05 - binary_accuracy: 1.0000
# Epoch 754/1000
# 1/1 - 0s - loss: 1.0322e-05 - binary_accuracy: 1.0000
# Epoch 755/1000
# 1/1 - 0s - loss: 1.0290e-05 - binary_accuracy: 1.0000
# Epoch 756/1000
# 1/1 - 0s - loss: 1.0260e-05 - binary_accuracy: 1.0000
# Epoch 757/1000
# 1/1 - 0s - loss: 1.0229e-05 - binary_accuracy: 1.0000
# Epoch 758/1000
# 1/1 - 0s - loss: 1.0197e-05 - binary_accuracy: 1.0000
# Epoch 759/1000
# 1/1 - 0s - loss: 1.0167e-05 - binary_accuracy: 1.0000
# Epoch 760/1000
# 1/1 - 0s - loss: 1.0136e-05 - binary_accuracy: 1.0000
# Epoch 761/1000
# 1/1 - 0s - loss: 1.0106e-05 - binary_accuracy: 1.0000
# Epoch 762/1000
# 1/1 - 0s - loss: 1.0075e-05 - binary_accuracy: 1.0000
# Epoch 763/1000
# 1/1 - 0s - loss: 1.0045e-05 - binary_accuracy: 1.0000
# Epoch 764/1000
# 1/1 - 0s - loss: 1.0016e-05 - binary_accuracy: 1.0000
# Epoch 765/1000
# 1/1 - 0s - loss: 9.9854e-06 - binary_accuracy: 1.0000
# Epoch 766/1000
# 1/1 - 0s - loss: 9.9548e-06 - binary_accuracy: 1.0000
# Epoch 767/1000
# 1/1 - 0s - loss: 9.9256e-06 - binary_accuracy: 1.0000
# Epoch 768/1000
# 1/1 - 0s - loss: 9.8964e-06 - binary_accuracy: 1.0000
# Epoch 769/1000
# 1/1 - 0s - loss: 9.8668e-06 - binary_accuracy: 1.0000
# Epoch 770/1000
# 1/1 - 0s - loss: 9.8375e-06 - binary_accuracy: 1.0000
# Epoch 771/1000
# 1/1 - 0s - loss: 9.8079e-06 - binary_accuracy: 1.0000
# Epoch 772/1000
# 1/1 - 0s - loss: 9.7793e-06 - binary_accuracy: 1.0000
# Epoch 773/1000
# 1/1 - 0s - loss: 9.7503e-06 - binary_accuracy: 1.0000
# Epoch 774/1000
# 1/1 - 0s - loss: 9.7214e-06 - binary_accuracy: 1.0000
# Epoch 775/1000
# 1/1 - 0s - loss: 9.6932e-06 - binary_accuracy: 1.0000
# Epoch 776/1000
# 1/1 - 0s - loss: 9.6646e-06 - binary_accuracy: 1.0000
# Epoch 777/1000
# 1/1 - 0s - loss: 9.6364e-06 - binary_accuracy: 1.0000
# Epoch 778/1000
# 1/1 - 0s - loss: 9.6077e-06 - binary_accuracy: 1.0000
# Epoch 779/1000
# 1/1 - 0s - loss: 9.5794e-06 - binary_accuracy: 1.0000
# Epoch 780/1000
# 1/1 - 0s - loss: 9.5509e-06 - binary_accuracy: 1.0000
# Epoch 781/1000
# 1/1 - 0s - loss: 9.5228e-06 - binary_accuracy: 1.0000
# Epoch 782/1000
# 1/1 - 0s - loss: 9.4953e-06 - binary_accuracy: 1.0000
# Epoch 783/1000
# 1/1 - 0s - loss: 9.4678e-06 - binary_accuracy: 1.0000
# Epoch 784/1000
# 1/1 - 0s - loss: 9.4405e-06 - binary_accuracy: 1.0000
# Epoch 785/1000
# 1/1 - 0s - loss: 9.4133e-06 - binary_accuracy: 1.0000
# Epoch 786/1000
# 1/1 - 0s - loss: 9.3858e-06 - binary_accuracy: 1.0000
# Epoch 787/1000
# 1/1 - 0s - loss: 9.3583e-06 - binary_accuracy: 1.0000
# Epoch 788/1000
# 1/1 - 0s - loss: 9.3316e-06 - binary_accuracy: 1.0000
# Epoch 789/1000
# 1/1 - 0s - loss: 9.3041e-06 - binary_accuracy: 1.0000
# Epoch 790/1000
# 1/1 - 0s - loss: 9.2779e-06 - binary_accuracy: 1.0000
# Epoch 791/1000
# 1/1 - 0s - loss: 9.2514e-06 - binary_accuracy: 1.0000
# Epoch 792/1000
# 1/1 - 0s - loss: 9.2248e-06 - binary_accuracy: 1.0000
# Epoch 793/1000
# 1/1 - 0s - loss: 9.1983e-06 - binary_accuracy: 1.0000
# Epoch 794/1000
# 1/1 - 0s - loss: 9.1716e-06 - binary_accuracy: 1.0000
# Epoch 795/1000
# 1/1 - 0s - loss: 9.1453e-06 - binary_accuracy: 1.0000
# Epoch 796/1000
# 1/1 - 0s - loss: 9.1185e-06 - binary_accuracy: 1.0000
# Epoch 797/1000
# 1/1 - 0s - loss: 9.0931e-06 - binary_accuracy: 1.0000
# Epoch 798/1000
# 1/1 - 0s - loss: 9.0670e-06 - binary_accuracy: 1.0000
# Epoch 799/1000
# 1/1 - 0s - loss: 9.0408e-06 - binary_accuracy: 1.0000
# Epoch 800/1000
# 1/1 - 0s - loss: 9.0152e-06 - binary_accuracy: 1.0000
# Epoch 801/1000
# 1/1 - 0s - loss: 8.9896e-06 - binary_accuracy: 1.0000
# Epoch 802/1000
# 1/1 - 0s - loss: 8.9637e-06 - binary_accuracy: 1.0000
# Epoch 803/1000
# 1/1 - 0s - loss: 8.9384e-06 - binary_accuracy: 1.0000
# Epoch 804/1000
# 1/1 - 0s - loss: 8.9131e-06 - binary_accuracy: 1.0000
# Epoch 805/1000
# 1/1 - 0s - loss: 8.8877e-06 - binary_accuracy: 1.0000
# Epoch 806/1000
# 1/1 - 0s - loss: 8.8627e-06 - binary_accuracy: 1.0000
# Epoch 807/1000
# 1/1 - 0s - loss: 8.8378e-06 - binary_accuracy: 1.0000
# Epoch 808/1000
# 1/1 - 0s - loss: 8.8129e-06 - binary_accuracy: 1.0000
# Epoch 809/1000
# 1/1 - 0s - loss: 8.7884e-06 - binary_accuracy: 1.0000
# Epoch 810/1000
# 1/1 - 0s - loss: 8.7636e-06 - binary_accuracy: 1.0000
# Epoch 811/1000
# 1/1 - 0s - loss: 8.7387e-06 - binary_accuracy: 1.0000
# Epoch 812/1000
# 1/1 - 0s - loss: 8.7142e-06 - binary_accuracy: 1.0000
# Epoch 813/1000
# 1/1 - 0s - loss: 8.6898e-06 - binary_accuracy: 1.0000
# Epoch 814/1000
# 1/1 - 0s - loss: 8.6656e-06 - binary_accuracy: 1.0000
# Epoch 815/1000
# 1/1 - 0s - loss: 8.6416e-06 - binary_accuracy: 1.0000
# Epoch 816/1000
# 1/1 - 0s - loss: 8.6176e-06 - binary_accuracy: 1.0000
# Epoch 817/1000
# 1/1 - 0s - loss: 8.5939e-06 - binary_accuracy: 1.0000
# Epoch 818/1000
# 1/1 - 0s - loss: 8.5700e-06 - binary_accuracy: 1.0000
# Epoch 819/1000
# 1/1 - 0s - loss: 8.5462e-06 - binary_accuracy: 1.0000
# Epoch 820/1000
# 1/1 - 0s - loss: 8.5222e-06 - binary_accuracy: 1.0000
# Epoch 821/1000
# 1/1 - 0s - loss: 8.4998e-06 - binary_accuracy: 1.0000
# Epoch 822/1000
# 1/1 - 0s - loss: 8.4758e-06 - binary_accuracy: 1.0000
# Epoch 823/1000
# 1/1 - 0s - loss: 8.4520e-06 - binary_accuracy: 1.0000
# Epoch 824/1000
# 1/1 - 0s - loss: 8.4290e-06 - binary_accuracy: 1.0000
# Epoch 825/1000
# 1/1 - 0s - loss: 8.4057e-06 - binary_accuracy: 1.0000
# Epoch 826/1000
# 1/1 - 0s - loss: 8.3828e-06 - binary_accuracy: 1.0000
# Epoch 827/1000
# 1/1 - 0s - loss: 8.3598e-06 - binary_accuracy: 1.0000
# Epoch 828/1000
# 1/1 - 0s - loss: 8.3363e-06 - binary_accuracy: 1.0000
# Epoch 829/1000
# 1/1 - 0s - loss: 8.3133e-06 - binary_accuracy: 1.0000
# Epoch 830/1000
# 1/1 - 0s - loss: 8.2919e-06 - binary_accuracy: 1.0000
# Epoch 831/1000
# 1/1 - 0s - loss: 8.2694e-06 - binary_accuracy: 1.0000
# Epoch 832/1000
# 1/1 - 0s - loss: 8.2454e-06 - binary_accuracy: 1.0000
# Epoch 833/1000
# 1/1 - 0s - loss: 8.2236e-06 - binary_accuracy: 1.0000
# Epoch 834/1000
# 1/1 - 0s - loss: 8.2019e-06 - binary_accuracy: 1.0000
# Epoch 835/1000
# 1/1 - 0s - loss: 8.1800e-06 - binary_accuracy: 1.0000
# Epoch 836/1000
# 1/1 - 0s - loss: 8.1580e-06 - binary_accuracy: 1.0000
# Epoch 837/1000
# 1/1 - 0s - loss: 8.1360e-06 - binary_accuracy: 1.0000
# Epoch 838/1000
# 1/1 - 0s - loss: 8.1137e-06 - binary_accuracy: 1.0000
# Epoch 839/1000
# 1/1 - 0s - loss: 8.0919e-06 - binary_accuracy: 1.0000
# Epoch 840/1000
# 1/1 - 0s - loss: 8.0696e-06 - binary_accuracy: 1.0000
# Epoch 841/1000
# 1/1 - 0s - loss: 8.0473e-06 - binary_accuracy: 1.0000
# Epoch 842/1000
# 1/1 - 0s - loss: 8.0254e-06 - binary_accuracy: 1.0000
# Epoch 843/1000
# 1/1 - 0s - loss: 8.0034e-06 - binary_accuracy: 1.0000
# Epoch 844/1000
# 1/1 - 0s - loss: 7.9821e-06 - binary_accuracy: 1.0000
# Epoch 845/1000
# 1/1 - 0s - loss: 7.9602e-06 - binary_accuracy: 1.0000
# Epoch 846/1000
# 1/1 - 0s - loss: 7.9391e-06 - binary_accuracy: 1.0000
# Epoch 847/1000
# 1/1 - 0s - loss: 7.9179e-06 - binary_accuracy: 1.0000
# Epoch 848/1000
# 1/1 - 0s - loss: 7.8967e-06 - binary_accuracy: 1.0000
# Epoch 849/1000
# 1/1 - 0s - loss: 7.8758e-06 - binary_accuracy: 1.0000
# Epoch 850/1000
# 1/1 - 0s - loss: 7.8543e-06 - binary_accuracy: 1.0000
# Epoch 851/1000
# 1/1 - 0s - loss: 7.8335e-06 - binary_accuracy: 1.0000
# Epoch 852/1000
# 1/1 - 0s - loss: 7.8127e-06 - binary_accuracy: 1.0000
# Epoch 853/1000
# 1/1 - 0s - loss: 7.7924e-06 - binary_accuracy: 1.0000
# Epoch 854/1000
# 1/1 - 0s - loss: 7.7713e-06 - binary_accuracy: 1.0000
# Epoch 855/1000
# 1/1 - 0s - loss: 7.7505e-06 - binary_accuracy: 1.0000
# Epoch 856/1000
# 1/1 - 0s - loss: 7.7302e-06 - binary_accuracy: 1.0000
# Epoch 857/1000
# 1/1 - 0s - loss: 7.7099e-06 - binary_accuracy: 1.0000
# Epoch 858/1000
# 1/1 - 0s - loss: 7.6901e-06 - binary_accuracy: 1.0000
# Epoch 859/1000
# 1/1 - 0s - loss: 7.6697e-06 - binary_accuracy: 1.0000
# Epoch 860/1000
# 1/1 - 0s - loss: 7.6495e-06 - binary_accuracy: 1.0000
# Epoch 861/1000
# 1/1 - 0s - loss: 7.6290e-06 - binary_accuracy: 1.0000
# Epoch 862/1000
# 1/1 - 0s - loss: 7.6091e-06 - binary_accuracy: 1.0000
# Epoch 863/1000
# 1/1 - 0s - loss: 7.5890e-06 - binary_accuracy: 1.0000
# Epoch 864/1000
# 1/1 - 0s - loss: 7.5689e-06 - binary_accuracy: 1.0000
# Epoch 865/1000
# 1/1 - 0s - loss: 7.5494e-06 - binary_accuracy: 1.0000
# Epoch 866/1000
# 1/1 - 0s - loss: 7.5300e-06 - binary_accuracy: 1.0000
# Epoch 867/1000
# 1/1 - 0s - loss: 7.5103e-06 - binary_accuracy: 1.0000
# Epoch 868/1000
# 1/1 - 0s - loss: 7.4909e-06 - binary_accuracy: 1.0000
# Epoch 869/1000
# 1/1 - 0s - loss: 7.4711e-06 - binary_accuracy: 1.0000
# Epoch 870/1000
# 1/1 - 0s - loss: 7.4514e-06 - binary_accuracy: 1.0000
# Epoch 871/1000
# 1/1 - 0s - loss: 7.4329e-06 - binary_accuracy: 1.0000
# Epoch 872/1000
# 1/1 - 0s - loss: 7.4135e-06 - binary_accuracy: 1.0000
# Epoch 873/1000
# 1/1 - 0s - loss: 7.3936e-06 - binary_accuracy: 1.0000
# Epoch 874/1000
# 1/1 - 0s - loss: 7.3748e-06 - binary_accuracy: 1.0000
# Epoch 875/1000
# 1/1 - 0s - loss: 7.3559e-06 - binary_accuracy: 1.0000
# Epoch 876/1000
# 1/1 - 0s - loss: 7.3370e-06 - binary_accuracy: 1.0000
# Epoch 877/1000
# 1/1 - 0s - loss: 7.3177e-06 - binary_accuracy: 1.0000
# Epoch 878/1000
# 1/1 - 0s - loss: 7.2988e-06 - binary_accuracy: 1.0000
# Epoch 879/1000
# 1/1 - 0s - loss: 7.2806e-06 - binary_accuracy: 1.0000
# Epoch 880/1000
# 1/1 - 0s - loss: 7.2616e-06 - binary_accuracy: 1.0000
# Epoch 881/1000
# 1/1 - 0s - loss: 7.2429e-06 - binary_accuracy: 1.0000
# Epoch 882/1000
# 1/1 - 0s - loss: 7.2246e-06 - binary_accuracy: 1.0000
# Epoch 883/1000
# 1/1 - 0s - loss: 7.2062e-06 - binary_accuracy: 1.0000
# Epoch 884/1000
# 1/1 - 0s - loss: 7.1878e-06 - binary_accuracy: 1.0000
# Epoch 885/1000
# 1/1 - 0s - loss: 7.1694e-06 - binary_accuracy: 1.0000
# Epoch 886/1000
# 1/1 - 0s - loss: 7.1509e-06 - binary_accuracy: 1.0000
# Epoch 887/1000
# 1/1 - 0s - loss: 7.1323e-06 - binary_accuracy: 1.0000
# Epoch 888/1000
# 1/1 - 0s - loss: 7.1153e-06 - binary_accuracy: 1.0000
# Epoch 889/1000
# 1/1 - 0s - loss: 7.0973e-06 - binary_accuracy: 1.0000
# Epoch 890/1000
# 1/1 - 0s - loss: 7.0780e-06 - binary_accuracy: 1.0000
# Epoch 891/1000
# 1/1 - 0s - loss: 7.0605e-06 - binary_accuracy: 1.0000
# Epoch 892/1000
# 1/1 - 0s - loss: 7.0428e-06 - binary_accuracy: 1.0000
# Epoch 893/1000
# 1/1 - 0s - loss: 7.0247e-06 - binary_accuracy: 1.0000
# Epoch 894/1000
# 1/1 - 0s - loss: 7.0070e-06 - binary_accuracy: 1.0000
# Epoch 895/1000
# 1/1 - 0s - loss: 6.9895e-06 - binary_accuracy: 1.0000
# Epoch 896/1000
# 1/1 - 0s - loss: 6.9717e-06 - binary_accuracy: 1.0000
# Epoch 897/1000
# 1/1 - 0s - loss: 6.9541e-06 - binary_accuracy: 1.0000
# Epoch 898/1000
# 1/1 - 0s - loss: 6.9366e-06 - binary_accuracy: 1.0000
# Epoch 899/1000
# 1/1 - 0s - loss: 6.9194e-06 - binary_accuracy: 1.0000
# Epoch 900/1000
# 1/1 - 0s - loss: 6.9022e-06 - binary_accuracy: 1.0000
# Epoch 901/1000
# 1/1 - 0s - loss: 6.8849e-06 - binary_accuracy: 1.0000
# Epoch 902/1000
# 1/1 - 0s - loss: 6.8675e-06 - binary_accuracy: 1.0000
# Epoch 903/1000
# 1/1 - 0s - loss: 6.8501e-06 - binary_accuracy: 1.0000
# Epoch 904/1000
# 1/1 - 0s - loss: 6.8337e-06 - binary_accuracy: 1.0000
# Epoch 905/1000
# 1/1 - 0s - loss: 6.8165e-06 - binary_accuracy: 1.0000
# Epoch 906/1000
# 1/1 - 0s - loss: 6.7994e-06 - binary_accuracy: 1.0000
# Epoch 907/1000
# 1/1 - 0s - loss: 6.7827e-06 - binary_accuracy: 1.0000
# Epoch 908/1000
# 1/1 - 0s - loss: 6.7660e-06 - binary_accuracy: 1.0000
# Epoch 909/1000
# 1/1 - 0s - loss: 6.7491e-06 - binary_accuracy: 1.0000
# Epoch 910/1000
# 1/1 - 0s - loss: 6.7322e-06 - binary_accuracy: 1.0000
# Epoch 911/1000
# 1/1 - 0s - loss: 6.7154e-06 - binary_accuracy: 1.0000
# Epoch 912/1000
# 1/1 - 0s - loss: 6.6984e-06 - binary_accuracy: 1.0000
# Epoch 913/1000
# 1/1 - 0s - loss: 6.6829e-06 - binary_accuracy: 1.0000
# Epoch 914/1000
# 1/1 - 0s - loss: 6.6665e-06 - binary_accuracy: 1.0000
# Epoch 915/1000
# 1/1 - 0s - loss: 6.6489e-06 - binary_accuracy: 1.0000
# Epoch 916/1000
# 1/1 - 0s - loss: 6.6329e-06 - binary_accuracy: 1.0000
# Epoch 917/1000
# 1/1 - 0s - loss: 6.6167e-06 - binary_accuracy: 1.0000
# Epoch 918/1000
# 1/1 - 0s - loss: 6.6004e-06 - binary_accuracy: 1.0000
# Epoch 919/1000
# 1/1 - 0s - loss: 6.5840e-06 - binary_accuracy: 1.0000
# Epoch 920/1000
# 1/1 - 0s - loss: 6.5678e-06 - binary_accuracy: 1.0000
# Epoch 921/1000
# 1/1 - 0s - loss: 6.5527e-06 - binary_accuracy: 1.0000
# Epoch 922/1000
# 1/1 - 0s - loss: 6.5366e-06 - binary_accuracy: 1.0000
# Epoch 923/1000
# 1/1 - 0s - loss: 6.5199e-06 - binary_accuracy: 1.0000
# Epoch 924/1000
# 1/1 - 0s - loss: 6.5043e-06 - binary_accuracy: 1.0000
# Epoch 925/1000
# 1/1 - 0s - loss: 6.4884e-06 - binary_accuracy: 1.0000
# Epoch 926/1000
# 1/1 - 0s - loss: 6.4729e-06 - binary_accuracy: 1.0000
# Epoch 927/1000
# 1/1 - 0s - loss: 6.4570e-06 - binary_accuracy: 1.0000
# Epoch 928/1000
# 1/1 - 0s - loss: 6.4411e-06 - binary_accuracy: 1.0000
# Epoch 929/1000
# 1/1 - 0s - loss: 6.4257e-06 - binary_accuracy: 1.0000
# Epoch 930/1000
# 1/1 - 0s - loss: 6.4099e-06 - binary_accuracy: 1.0000
# Epoch 931/1000
# 1/1 - 0s - loss: 6.3943e-06 - binary_accuracy: 1.0000
# Epoch 932/1000
# 1/1 - 0s - loss: 6.3789e-06 - binary_accuracy: 1.0000
# Epoch 933/1000
# 1/1 - 0s - loss: 6.3637e-06 - binary_accuracy: 1.0000
# Epoch 934/1000
# 1/1 - 0s - loss: 6.3485e-06 - binary_accuracy: 1.0000
# Epoch 935/1000
# 1/1 - 0s - loss: 6.3331e-06 - binary_accuracy: 1.0000
# Epoch 936/1000
# 1/1 - 0s - loss: 6.3176e-06 - binary_accuracy: 1.0000
# Epoch 937/1000
# 1/1 - 0s - loss: 6.3023e-06 - binary_accuracy: 1.0000
# Epoch 938/1000
# 1/1 - 0s - loss: 6.2880e-06 - binary_accuracy: 1.0000
# Epoch 939/1000
# 1/1 - 0s - loss: 6.2729e-06 - binary_accuracy: 1.0000
# Epoch 940/1000
# 1/1 - 0s - loss: 6.2571e-06 - binary_accuracy: 1.0000
# Epoch 941/1000
# 1/1 - 0s - loss: 6.2426e-06 - binary_accuracy: 1.0000
# Epoch 942/1000
# 1/1 - 0s - loss: 6.2275e-06 - binary_accuracy: 1.0000
# Epoch 943/1000
# 1/1 - 0s - loss: 6.2127e-06 - binary_accuracy: 1.0000
# Epoch 944/1000
# 1/1 - 0s - loss: 6.1979e-06 - binary_accuracy: 1.0000
# Epoch 945/1000
# 1/1 - 0s - loss: 6.1828e-06 - binary_accuracy: 1.0000
# Epoch 946/1000
# 1/1 - 0s - loss: 6.1686e-06 - binary_accuracy: 1.0000
# Epoch 947/1000
# 1/1 - 0s - loss: 6.1535e-06 - binary_accuracy: 1.0000
# Epoch 948/1000
# 1/1 - 0s - loss: 6.1391e-06 - binary_accuracy: 1.0000
# Epoch 949/1000
# 1/1 - 0s - loss: 6.1249e-06 - binary_accuracy: 1.0000
# Epoch 950/1000
# 1/1 - 0s - loss: 6.1103e-06 - binary_accuracy: 1.0000
# Epoch 951/1000
# 1/1 - 0s - loss: 6.0959e-06 - binary_accuracy: 1.0000
# Epoch 952/1000
# 1/1 - 0s - loss: 6.0814e-06 - binary_accuracy: 1.0000
# Epoch 953/1000
# 1/1 - 0s - loss: 6.0670e-06 - binary_accuracy: 1.0000
# Epoch 954/1000
# 1/1 - 0s - loss: 6.0524e-06 - binary_accuracy: 1.0000
# Epoch 955/1000
# 1/1 - 0s - loss: 6.0384e-06 - binary_accuracy: 1.0000
# Epoch 956/1000
# 1/1 - 0s - loss: 6.0242e-06 - binary_accuracy: 1.0000
# Epoch 957/1000
# 1/1 - 0s - loss: 6.0098e-06 - binary_accuracy: 1.0000
# Epoch 958/1000
# 1/1 - 0s - loss: 5.9959e-06 - binary_accuracy: 1.0000
# Epoch 959/1000
# 1/1 - 0s - loss: 5.9819e-06 - binary_accuracy: 1.0000
# Epoch 960/1000
# 1/1 - 0s - loss: 5.9678e-06 - binary_accuracy: 1.0000
# Epoch 961/1000
# 1/1 - 0s - loss: 5.9538e-06 - binary_accuracy: 1.0000
# Epoch 962/1000
# 1/1 - 0s - loss: 5.9396e-06 - binary_accuracy: 1.0000
# Epoch 963/1000
# 1/1 - 0s - loss: 5.9261e-06 - binary_accuracy: 1.0000
# Epoch 964/1000
# 1/1 - 0s - loss: 5.9123e-06 - binary_accuracy: 1.0000
# Epoch 965/1000
# 1/1 - 0s - loss: 5.8985e-06 - binary_accuracy: 1.0000
# Epoch 966/1000
# 1/1 - 0s - loss: 5.8848e-06 - binary_accuracy: 1.0000
# Epoch 967/1000
# 1/1 - 0s - loss: 5.8712e-06 - binary_accuracy: 1.0000
# Epoch 968/1000
# 1/1 - 0s - loss: 5.8575e-06 - binary_accuracy: 1.0000
# Epoch 969/1000
# 1/1 - 0s - loss: 5.8440e-06 - binary_accuracy: 1.0000
# Epoch 970/1000
# 1/1 - 0s - loss: 5.8301e-06 - binary_accuracy: 1.0000
# Epoch 971/1000
# 1/1 - 0s - loss: 5.8165e-06 - binary_accuracy: 1.0000
# Epoch 972/1000
# 1/1 - 0s - loss: 5.8037e-06 - binary_accuracy: 1.0000
# Epoch 973/1000
# 1/1 - 0s - loss: 5.7901e-06 - binary_accuracy: 1.0000
# Epoch 974/1000
# 1/1 - 0s - loss: 5.7762e-06 - binary_accuracy: 1.0000
# Epoch 975/1000
# 1/1 - 0s - loss: 5.7632e-06 - binary_accuracy: 1.0000
# Epoch 976/1000
# 1/1 - 0s - loss: 5.7500e-06 - binary_accuracy: 1.0000
# Epoch 977/1000
# 1/1 - 0s - loss: 5.7366e-06 - binary_accuracy: 1.0000
# Epoch 978/1000
# 1/1 - 0s - loss: 5.7233e-06 - binary_accuracy: 1.0000
# Epoch 979/1000
# 1/1 - 0s - loss: 5.7100e-06 - binary_accuracy: 1.0000
# Epoch 980/1000
# 1/1 - 0s - loss: 5.6974e-06 - binary_accuracy: 1.0000
# Epoch 981/1000
# 1/1 - 0s - loss: 5.6840e-06 - binary_accuracy: 1.0000
# Epoch 982/1000
# 1/1 - 0s - loss: 5.6711e-06 - binary_accuracy: 1.0000
# Epoch 983/1000
# 1/1 - 0s - loss: 5.6585e-06 - binary_accuracy: 1.0000
# Epoch 984/1000
# 1/1 - 0s - loss: 5.6456e-06 - binary_accuracy: 1.0000
# Epoch 985/1000
# 1/1 - 0s - loss: 5.6328e-06 - binary_accuracy: 1.0000
# Epoch 986/1000
# 1/1 - 0s - loss: 5.6198e-06 - binary_accuracy: 1.0000
# Epoch 987/1000
# 1/1 - 0s - loss: 5.6069e-06 - binary_accuracy: 1.0000
# Epoch 988/1000
# 1/1 - 0s - loss: 5.5939e-06 - binary_accuracy: 1.0000
# Epoch 989/1000
# 1/1 - 0s - loss: 5.5818e-06 - binary_accuracy: 1.0000
# Epoch 990/1000
# 1/1 - 0s - loss: 5.5690e-06 - binary_accuracy: 1.0000
# Epoch 991/1000
# 1/1 - 0s - loss: 5.5559e-06 - binary_accuracy: 1.0000
# Epoch 992/1000
# 1/1 - 0s - loss: 5.5435e-06 - binary_accuracy: 1.0000
# Epoch 993/1000
# 1/1 - 0s - loss: 5.5311e-06 - binary_accuracy: 1.0000
# Epoch 994/1000
# 1/1 - 0s - loss: 5.5184e-06 - binary_accuracy: 1.0000
# Epoch 995/1000
# 1/1 - 0s - loss: 5.5059e-06 - binary_accuracy: 1.0000
# Epoch 996/1000
# 1/1 - 0s - loss: 5.4933e-06 - binary_accuracy: 1.0000
# Epoch 997/1000
# 1/1 - 0s - loss: 5.4812e-06 - binary_accuracy: 1.0000
# Epoch 998/1000
# 1/1 - 0s - loss: 5.4688e-06 - binary_accuracy: 1.0000
# Epoch 999/1000
# 1/1 - 0s - loss: 5.4565e-06 - binary_accuracy: 1.0000
# Epoch 1000/1000
# 1/1 - 0s - loss: 5.4445e-06 - binary_accuracy: 1.0000
# [[0.]
#  [1.]
#  [1.]
#  [0.]]
