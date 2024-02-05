import matplotlib.pyplot as plt


def plot_graphs(loss_train, loss_valid, accuracy, f1, title='Train info'):
    plt.figure(figsize=(7, 9))

    plt.subplot(2, 1, 1)
    plt.plot(loss_train, label='Loss train', color='blue')
    plt.plot(loss_valid, label='Loss valid', color='red')
    plt.title(title)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(accuracy, label='Accuracy', color='orange')
    plt.plot(f1, label='F1', color='green')
    plt.legend()

    plt.tight_layout()
    plt.show()


def create_graphs(train_file=r'D:\repos\ortho_classification\train_info.txt',
                  accuracy_f1_file=r'D:\repos\ortho_classification\accuracy_precision_recall_F1_info.txt'):
    with open(train_file, 'r') as f:
        train_loss = []
        valid_loss = []
        for line in f:
            if line.find('Train Loss') != -1 and line.find('Valid Loss') != -1:
                train_loss.append(float(line.split()[4]))
                valid_loss.append(float(line.split()[7]))

    with open(accuracy_f1_file, 'r') as f:
        accuracy = []
        f1 = []
        for line in f:
            accuracy.append(float(line.split()[3]))
            f1.append(float(line.split()[-1]))

    plot_graphs(train_loss, valid_loss, accuracy, f1)


if __name__ == '__main__':
    create_graphs()
