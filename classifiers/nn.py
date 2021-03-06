from numpy import genfromtxt
from utils.nn import NeuralNetwork


def train():
    training_full_dataset = genfromtxt("data/training_spam.csv", delimiter=',')
    test_full_dataset = genfromtxt("data/testing_spam.csv", delimiter=',')
    training_dataset = training_full_dataset[:, 1:].T
    training_labels = training_full_dataset[:, 0].T
    test_dataset = test_full_dataset[:, 1:].T
    test_labels = test_full_dataset[:, 0].T
    learning_rate = 1

    layer_settings = [{'nodes': 54, 'learning_rate': learning_rate},
                      {'nodes': 3, 'learning_rate': learning_rate},
                      {'nodes': 1, 'learning_rate': learning_rate}]
    nn = NeuralNetwork(layer_settings=layer_settings)
    nn.set_training_data(dataset=training_dataset,
                         labels=training_labels)
    nn.train(epochs=3000,
             epoch_size=0.2)
    nn.predict(test_dataset)
    acc = nn.classification_accuracy(dataset=test_dataset,
                                     labels=test_labels).round(3)
    nn.save_weights_and_biases("data/weights_and_bias.json")
    print(acc)


def classify(data):
    layer_settings = [{'nodes': 54},
                      {'nodes': 3},
                      {'nodes': 1}]
    nn = NeuralNetwork(layer_settings=layer_settings)
    nn.load_weights_and_biases("data/weights_and_bias.json")
    return nn.predict(data).round(0)


if __name__ == '__main__':
    test_dataset = genfromtxt("data/testing_spam.csv", delimiter=',')[:, 1:]
    a = classify(test_dataset)
    print(a)
