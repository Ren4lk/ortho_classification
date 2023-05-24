import matplotlib.pyplot as plt


with open('ortho_classification/accuracy_info.txt', 'r') as f:
    accuracies = []
    for line in f:
        accuracies.append(float(line.split()[3]))

    plt.plot(accuracies, 'b')
    plt.show()
