import mlnn as mlnn

# The majority function f(x, y, z) returns the majority value of the three boolean inputs.
# E.g., f(0, 1, 0) = 0 and f(1, 1, 0) = 1.
# Following is the program, which uses mlnn.py to create a neural network with the architecture (3, 4, 1)
# and train it with the definition of the majority function using proper epoch number
# and learning rate (assuming the batch size is 1).

if __name__ == '__main__':

    train_X = mlnn.np.array([
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [1, 1, 1]])

    train_Y =  mlnn.np.expand_dims(mlnn.np.array([0, 0, 0, 0, 1, 1, 1, 1]), axis=1)
   
    net = mlnn.Network([3, 4, 1])
    
    print('weight shapes:', [w.shape for w in net.weights])
    print('weights:', [w for w in net.weights])
        
    net.SGD(train_X, train_Y, 20, 1, 11)
    
    #for a in train_X: print(a, net.feedforward(a))
    for x, y in zip(train_X, net.feedforward(train_X)): print(x, y)

