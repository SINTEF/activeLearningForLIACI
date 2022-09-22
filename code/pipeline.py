from data import get_data
from model import model_create


def main():
    model = model_create()

    i = get_data()
    print(i)
    print(model)
    

if __name__ == "__main__":
    main()