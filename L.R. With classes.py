import pandas as pd #bibliteca para trabalhar com dados tabulados
import matplotlib.pyplot as plt #biblioteca para visualizar os dados
from sklearn.linear_model import LinearRegression

class MinhaIA:
    def __init__(self):
        self.nome = "Artemis"
        print('Inteligência artificial chamada com sucesso')
        self.comando = Comando()

    def imprime(self):
        print(f'Olá, meu nome é {self.nome} e irei ajudar você a calcular regressão linear.')

class Comando:
    def __init__(self):
        self.d = {'X': [], 'Y': []}

    def calcular(self):
        n = int(input('Informe quantos valores irá utilizar: '))
        for i in range(n):
            valorX = float(input('Informe o valor de x: '))
            valorY = float(input('Informe o valor do y correspondente: '))
            self.d['X'].append(valorX)
            self.d['Y'].append(valorY)

        dados = pd.DataFrame(self.d)

        dados['y_reta'] = dados.X

        fig, ax = plt.subplots()
        ax.scatter(dados.X, dados.Y)
        ax.scatter(dados.X, dados.y_reta) #scatter é para traçar os pontos
        ax.plot(dados.X, dados.y_reta, '--g')


        reg = LinearRegression().fit(dados.X.values.reshape(-1, 1), dados.Y)
        a = reg.coef_
        b = reg.intercept_

        x = dados.X.values
        y = a * x + b
        ax.plot(x, y)  # Reta gerada pela regressão linear


        # Chama o método de erros
        self.erros(dados, reg)

    def erros(self, dados, reg):
        opcao = input('Deseja também obter uma tabela dos dados e dos erros? (sim/não): ')

        plt.show
        if opcao.lower() == 'sim':
            dados['y_pred'] = reg.predict(dados.X.values.reshape(-1, 1))
            dados['erro_reta'] = (dados.Y - dados.y_reta)**2
            dados['erro_pred'] = (dados.Y - dados.y_pred)**2
            print(dados)  # Exibe a tabela com os dados e os erros

# Execução
artemis = MinhaIA()
artemis.imprime()
artemis.comando.calcular()
