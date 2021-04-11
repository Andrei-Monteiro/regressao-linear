import numpy as np

def compute_cost(theta_0, theta_1, data):
    """
    Calcula o erro quadratico medio
    
    Args:
        theta_0 (float): intercepto da reta 
        theta_1 (float): inclinacao da reta
        data (np.array): matriz com o conjunto de dados, x na coluna 0 e y na coluna 1
    
    Retorna:
        float: o erro quadratico medio
    """
    ### SEU CODIGO AQUI
    x = np.array(data[:,46])
    y = np.array(data[:,80])
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]
    max=np.amax(x)
    min=np.amin(x)
    total_cost=0
    for i in range(0,len(x)):
      xi=(x[i]-min)/(max-min)
      total_cost= total_cost+ ((theta_0+ theta_1*xi)-y[i])**2

    total_cost = total_cost/len(x)
    return total_cost

def step_gradient(theta_0_current, theta_1_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo
    
    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo 
    
    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """
    x = np.array(data[:,46])
    y = np.array(data[:,80])
    x = x[np.logical_not(np.isnan(x))]
    y = y[np.logical_not(np.isnan(y))]
    max=np.amax(x)
    min=np.amin(x)
    derivada_0=0
    derivada_1=0
    for i in range(0,len(x)):
      xi=(x[i]-min)/(max-min)
      derivada_0= derivada_0+ ((theta_0_current+ theta_1_current*xi)-y[i])
      derivada_1=derivada_1 + (((theta_0_current+ theta_1_current*xi)-y[i])*xi)

    derivada_0=2* derivada_0/len(x)
    derivada_1=2* derivada_1/len(x)
    theta_0_updated = theta_0_current- derivada_0*alpha
    theta_1_updated = theta_1_current- derivada_1*alpha
    return theta_0_updated, theta_1_updated

def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    """executa a descida do gradiente
    
    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0 
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar
    
    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    
    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    
    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha=0.0001)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        
    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]

data = np.genfromtxt('house_prices_train.csv', delimiter=',')
theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(data, starting_theta_0=0, starting_theta_1=0, learning_rate=0, num_iterations=100)
#Imprimir parâmetros otimizados
print ('Theta_0 otimizado: ', theta_0)
print ('Theta_1 otimizado: ', theta_1)
#Imprimir erro com os parâmetros otimizados
print ('Custo minimizado: ', compute_cost(theta_0, theta_1, data))