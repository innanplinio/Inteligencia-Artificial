import random
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("dataset.csv", delimiter=',')
train, test = train_test_split(df, test_size=0.4)

train_x = train.iloc[:, 48:57].values.tolist()
train_y = train['exit'].tolist()

test_x = test.iloc[:, 48:57].values.tolist()
test_y = test['exit'].tolist()


class Perceptron:

    # Inicializacao do objeto Perceptron
    def __init__(self, sample, exit, learn_rate=0.001, epoch_number=1000, bias=-1):
        self.sample = sample
        self.exit = exit
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.number_sample = len(sample)
        self.col_sample = len(sample[0])
        self.weight = []

    # Funcao de Treinamento do Perceptron (Metodo Gradiente Descendente)
    def trannig(self):
        for sample in self.sample:
            sample.insert(0, self.bias)

        # Inicializa os pesos w aleatoriamente
        for i in range(self.col_sample):
            self.weight.append(random.random())

        # Insere peso da entrada de polarizacao(bias)
        self.weight.insert(0, self.bias)

        epoch_count = 0

        # Metodo do Gradiente Descendente para ajuste dos pesos do Perceptron
        while True:
            erro = False
            for i in range(self.number_sample):
                u = 0
                for j in range(self.col_sample + 1):
                    u += self.weight[j] * self.sample[i][j]
                y = self.sign(u)
                if y != self.exit[i]:
                    aux_erro = self.exit[i] - y
                    for j in range(self.col_sample + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * aux_erro * self.sample[i][j]
                    erro = True
            print('Epoca: \n', epoch_count)
            epoch_count = epoch_count + 1
            # Se parada porepocas ou erro
            if erro == False or epoch_count > self.epoch_number :
                print(('\nEpocas:\n', epoch_count))
                print('------------------------\n')
                break

    def sort(self, sample):
        sample.insert(0, self.bias)
        u = 0
        for i in range(self.col_sample + 1):
            u += self.weight[i] * sample[i]

        y = self.sign(u)

        if y == 0:
            #print(('Sample: ', sample))
            return 0
        else:
            #print(('Sample: ', sample))
            return 1

    # Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else 0


# Inicializa o Perceptron
network = Perceptron(sample=train_x, exit=train_y)

# Chamada ao treinamento
network.trannig()

cont = 0
acerto0 = 0
acerto1 = 0
erro0 = 0
erro1 = 0
while cont < len(test_y):
    if network.sort(test_x[cont]) == test_y[cont]:
        if test_y[cont] == 0:
            acerto0 += 1
        else:
            acerto1 += 1
    else:
        if test_y[cont] == 0:
            erro0 += 1
        else:
            erro1 += 1
    cont += 1

print('0&0 1&1 1&0 0&1')
print(acerto0, acerto1, erro0, erro1)
print(len(test_y))

cont = 0
acerto = 0
erro = 0

while cont < len(test_y):
    if network.sort(test_x[cont]) == test_y[cont]:
        acerto += 1
    else:
        erro += 1
    cont += 1

print(acerto)
print(erro)
print(len(test_y))

