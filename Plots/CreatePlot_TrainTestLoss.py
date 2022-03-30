import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main():

    df_train_loss = pd.read_csv('run-.-tag-Train loss.csv', sep=',')
    train_loss = df_train_loss["Value"]

    df_test_loss = pd.read_csv('run-.-tag-Test loss.csv', sep=',')
    test_loss = df_test_loss["Value"]

    x = np.arange(len(train_loss))
    
    plt.plot(x, train_loss, label="Train loss", color="r")
    plt.plot(x, test_loss, label="Test loss", color="b")

    plt.legend()
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig("TrainTestLoss.png")
    plt.show()
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")