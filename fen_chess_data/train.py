
import pandas as pd
import numpy as np
import time
import tensorflow as tf

from models import build_model
from prepro import load_dataset
from utils import callback_test_error

# hyperparameters
N_ITER = 50000
BATCH_SIZE = 64

LR = 0.0001
DECAY = 0.00001
BETA1 = 0.5
BETA2 = 0.9
EPSILON = 1e-06

start_time = time.strftime("%Y%m%d%H%M%S")

def train():

    d = 32

    # load data
    X_train, X_test, y_train, y_test, X_train_names, X_test_names = load_dataset(d)

    LEN_DF = X_train.shape[0]

    print(X_train.shape, y_train.shape)

    # load model
    input_shape = (d, d, 3)
    model = build_model(input_shape)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LR,
            clipvalue=1.,
            beta_1=BETA1,
            beta_2=BETA2,
            epsilon=EPSILON
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        metrics=["accuracy"]
    )

    print(model.summary())

    losses = np.zeros(2)
    loss_iter = 100

    max_val_acc = 0.00

    for i in range(N_ITER):

        idx = np.random.randint(LEN_DF, size=BATCH_SIZE)
        labels, imgs = y_train[idx], X_train[idx]

        loss = model.train_on_batch(
            x=imgs,
            y=labels
        )

        losses += loss

        if i % loss_iter == 0:

            preds = model.predict(np.float32(X_test))
            test_acc = callback_test_error(y_test, preds)

            print( 
                i, 
                'tr loss {0:.4f}'.format(losses[0] / loss_iter), 
                'tr acc {0:.4f}'.format(losses[1] / loss_iter), 
                'val acc {0:.4f}'.format(test_acc), 
            )
            
            losses = np.zeros(2)

            if test_acc > max_val_acc:
                model.save('models/model_best.h5')
                print(f'New best model saved! {i} iter')

                max_val_acc = test_acc


if __name__ == "__main__":
    
    train()
