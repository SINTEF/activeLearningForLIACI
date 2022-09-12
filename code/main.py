import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense
import numpy as np
# from tf.keras import Model




def main():
    mobile = MobileNet()

    print("Code starts here\r\n=======================\r\n")
    x = mobile.layers[-6].output
    output = Dense(units=2, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=output)æ
    print(x)

if __name__ == "__main__":
    main()