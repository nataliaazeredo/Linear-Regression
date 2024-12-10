print('LINEAR REGRESSION WITH ONE VARIABLE')

import matplotlib.pyplot as plt #biblioteca para visualizar os dados
import numpy as np

x = [1,2,3,4,5]
y = [1.3, 1.8, 3.5, 4, 4.6]


def regressao(x,y):
  sum_x = 0
  sum_y =0
  sum_x2 =0
  sum_xy =0

  n = len(x)
  for i in range(n):
    sum_x += x[i]
    sum_y += y[i]
    sum_x2 += x[i] ** 2
    sum_xy += x[i] * y[i]

  m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
  b = (sum_x*sum_xy-sum_y*sum_x2)/((sum_x)**2-n*sum_x2)

  return m,b


def previsao(x,y,a,b):
  y_pred = [a * x[i] + b for i in range(len(x))]

  return y_pred


def MSE(x,y,a,b):
  custo = 0
  y_custo = lambda x,a,b: a*x+b
  for i in range(len(x)):
    custo += (y_custo(x[i], a, b)-y[i])**2

  return custo/(2*len(x))

def step(x,y,a,b,l_r): #Atualização dos parametros
  erro_b = 0
  erro_a = 0
  m = len(x)
  y_custo = lambda x,a,b: a*x+b
  for i in range(len(x)):

    erro_b += y_custo(x[i],a,b)-y[i]
    erro_a += (y_custo(x[i],a,b)-y[i])*x[i]

  b2 = b-l_r*(1/m)*erro_b
  a2 = a-l_r*(1/m)*erro_a

  return a2, b2


def gradiente(x,y,a, b,l_r,epoch):
  custo = []
  for i in range(epoch):
    a, b = step(x,y,a,b,l_r)
    custo.append(MSE(x,y,a,b))

  return a,b,custo


def plot_line(x,y):
  a,b = regressao(x,y)
  plt.scatter(x,y,color='red', marker='*')
  plt.plot(x,previsao(x,y,a,b), 'b')
  plt.title('Regressão Linear')
  plt.xlabel('x')
  plt.ylabel('y')


def main(x, y):
    epoch = 50
    l_r = 0.15

    a = 0
    b = 0
    a, b, custo = gradiente(x, y, a, b, l_r, epoch)

    plt.figure(figsize=(12, 6)) #Ajusta o tamanho da figura
    plt.subplot(1, 2, 1) #Cria o gráfico na parte esquerda
    plot_line(x, y)

    plt.subplot(1, 2, 2) #Cria o gráfico na parte direita
    plt.plot(np.arange(epoch), custo, 'r')
    plt.xlabel('Interações')
    plt.ylabel('Custo (MSE)')
    plt.title('MSE Vs. EPOCH')

    plt.tight_layout()
    plt.show()

main(x,y)
