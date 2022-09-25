from termios import N_MOUSE
from data import load_from_coco, get_cats
from model import mobilenet_create, model_create, summarize_diagnostics, train
import matplotlib.pyplot as plt


def main():
    n_imgs = 0
    epochs = 100
    n_cats = len(get_cats())
    lr = 2e-4
    X, Y = load_from_coco(n_imgs=n_imgs)
    # for a in X:
    #     plt.imshow(a)
    #     plt.show()
    mobilenet = mobilenet_create()

    res = mobilenet.predict(X) # Extract Features from mobilenet
    # print(res.shape)
    model = model_create(res.shape, n_cats=n_cats, lr=lr)

    h, e = train(
        model,
        res,
        Y,
        epochs=epochs
    )
    summarize_diagnostics(h, e)
    

    # mobilenet.summary()
    # model.summary()
    # rm = model.predict(res)
    # print(rm.shape)


if __name__ == "__main__":
    main()