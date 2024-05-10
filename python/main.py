import time, sys
import numpy as np
from nn import NeuralNet
from dataset import TRAIN_DATA, TEST_DATA


NN_model = NeuralNet(784, [50], 10)
#epoch_histogram = list()

def train_network(learning_rate, epochs, batch_size):
    #current_epoch_loss = list()
    N = len(TRAIN_DATA.images)
    for epoch in range(epochs):
        start = time.time()
        # Shuffle the training data (optional but recommended)
        permutation = np.random.permutation(N)
        X_train_shuffled = TRAIN_DATA.images[permutation]
        Y_train_shuffled = TRAIN_DATA.labels[permutation]
        
        loss_sum = 0
        count = 0
        for i in range(0, N, batch_size):
            # Mini-batch extraction
            X_batch = X_train_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]

            # Train batch      
            loss_sum += NN_model.train(X_batch, Y_batch, learning_rate) * batch_size
            #if i % (batch_size * 4) == 0:
            #    current_epoch_loss.append(loss_sum/count)

        loss = loss_sum/N
        end = time.time()
        print('{:> 4}: took {:> 2.2}s\tMean loss: {:> 2.4}'.format(epoch, end-start, loss))
        #epoch_histogram.append({
        #    'mean_loss': loss,
        #    'model': NN_model.clone(),
        #    'learning_rate': learning_rate,
        #    'batch_size': batch_size,
        #    'epoch': epoch,
        #    'elapsed': end-start,
        #    'epoch_loss': current_epoch_loss,
        #})
        #current_epoch_loss = list()

if __name__ == "__main__":
    train_network(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
    #train_network(0.01, 200, 128)    