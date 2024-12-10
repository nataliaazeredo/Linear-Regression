import pandas as pd #bibliteca para trabalhar com dados tabulados
import matplotlib.pyplot as plt #biblioteca para visualizar os dados
import numpy as np

#dados de treinamento
preco = [1200, 50000,30000]
banheiro = [1,4,2]
sala =[0,1,2]
quarto =[1,3,4]
bias = [0,0,0]

Y = (np.array(preco))
print(Y)

X = np.array([bias,banheiro, sala, quarto]).T
print(X)

def previsao(X,Y,theta):
  Y_pred = np.dot(X, theta)
  return Y_pred

def MSE(X, Y, theta):
    m = len(Y)
    Y_pred = np.dot(X, theta) #produto escalar
    MSE = (1/(2*m)) * np.sum((Y_pred - Y)**2)
    return MSE


def gradiente(X, Y, theta, l_r, epoch):
    m = len(Y)
    custo = []

    for i in range(epoch):

        Y_pred = np.dot(X, theta)
        erro = Y_pred - Y
        gradiente = (1/m) * np.dot(X.T, erro) #Gradiente na forma matricial
        theta = theta - l_r * gradiente #atualização do parametro
        custo.append(MSE(X, Y, theta))

    return theta, custo

def plot(X,Y, theta, custo):
  plt.scatter(X,Y,color='red', marker='*')
  plt.plot(X,previsao(X,Y,theta), 'b')
  plt.title('Regressão Linear com multiplas variaveis')
  plt.xlabel('x')
  plt.ylabel('y')

def main(X,Y):

  theta = np.zeros(X.shape[1])
  epoch = 1000
  learning_rate = 0.01
  theta_final, custo = gradiente(X, Y, theta, learning_rate, epoch)

  print(f'Theta final: {theta_final}')

  plt.figure(figsize=(12, 6)) #Ajusta o tamanho da figura
  plt.subplot(1, 2, 1) #Cria o gráfico na parte esquerda

  plt.subplot(1, 2, 2) #Cria o gráfico na parte direita
  plt.plot(np.arange(epoch), custo, 'r')
  plt.xlabel('Interações')
  plt.ylabel('Custo (MSE)')
  plt.title('MSE Vs. EPOCH')
  plt.tight_layout()
  plt.show()

main(X,Y)
