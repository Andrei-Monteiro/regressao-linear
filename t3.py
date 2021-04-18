import numpy as np
import sys


def compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5,data):
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
    x1 = np.array(data[:,46])
    y = np.array(data[:,80])

    x2 = np.array(data[:,17])
    x3 = np.array(data[:,18])
    x4 = np.array(data[:,50])
    x5 = np.array(data[:,19])

    x1 = np.delete(x1, 0)
    x2 = np.delete(x2, 0)
    x3 = np.delete(x3, 0)
    x4 = np.delete(x4, 0)
    x5 = np.delete(x5, 0)
    y = np.delete(y, 0)
    tam=len(x1)
    max_x1=np.amax(x1)
    min_x1=np.amin(x1)
    max_x2=np.amax(x2)
    min_x2=np.amin(x2)
    max_x3=np.amax(x3)
    min_x3=np.amin(x3)
    max_x4=np.amax(x4)
    min_x4=np.amin(x4)
    max_x5=np.amax(x5)
    min_x5=np.amin(x5)

    total_cost=0
    if atributos==1:
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            h0=theta_0+ theta_1*x1_aux
            total_cost+=(y[i]-(h0))**2
    if (atributos==2):
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            x2_aux=(x2[i]-min_x2)/(max_x2-min_x2)
            h0=theta_0+ theta_1*x1_aux+theta_2*x2_aux
            total_cost+=(y[i]-(h0))**2
    if(atributos==5):
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            x2_aux=(x2[i]-min_x2)/(max_x2-min_x2)
            x3_aux=(x3[i]-min_x3)/(max_x3-min_x3)
            x4_aux=(x4[i]-min_x4)/(max_x4-min_x4)
            x5_aux=(x5[i]-min_x5)/(max_x5-min_x5)
            h0=theta_0+ theta_1*x1_aux+theta_2*x2_aux+theta_3*x3_aux+theta_4*x4_aux+theta_5_*x5_aux
            total_cost+=(y[i]-(h0))**2

    total_cost = total_cost/tam
    return total_cost

def step_gradient(theta_0_current, theta_1_current,theta_2_current, theta_3_current,theta_4_current,theta_5_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo
    
    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo 
    
    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """
    x1 = np.array(data[:,46])
    y = np.array(data[:,80])
    x2 = np.array(data[:,17])
    x3 = np.array(data[:,18])
    x4 = np.array(data[:,50])
    x5 = np.array(data[:,19])

    x1 = np.delete(x1, 0)
    x2 = np.delete(x2, 0)
    x3 = np.delete(x3, 0)
    x4 = np.delete(x4, 0)
    x5 = np.delete(x5, 0)
    y = np.delete(y, 0)
    tam=len(x1)

    max_x1=np.amax(x1)
    min_x1=np.amin(x1)
    max_x2=np.amax(x2)
    min_x2=np.amin(x2)
    max_x3=np.amax(x3)
    min_x3=np.amin(x3)
    max_x4=np.amax(x4)
    min_x4=np.amin(x4)
    max_x5=np.amax(x5)
    min_x5=np.amin(x5)

    soma_0=0
    soma_1=0

    if(atributos==1):
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            h0=theta_0_current+ theta_1_current*x1_aux
            soma_0+= h0 - y[i]
            soma_1+= (h0 - y[i]) * x1_aux
        derivada_0=2* soma_0/tam
        derivada_1=2* soma_1/tam
        theta_0_updated = theta_0_current- derivada_0*alpha
        theta_1_updated = theta_1_current- derivada_1*alpha
        return theta_0_updated, theta_1_updated

    if(atributos==2):
        soma_2=0
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            x2_aux=(x2[i]-min_x2)/(max_x2-min_x2)
            h0=theta_0_current+ theta_1_current*x1_aux+theta_2_current*x2_aux
            soma_0+= h0 - y[i]
            soma_1+= (h0 - y[i]) * x1_aux
            soma_2+= (h0 - y[i]) * x2_aux
        derivada_0=2* soma_0/tam
        derivada_1=2* soma_1/tam
        derivada_2=2* soma_2/tam
        theta_0_updated = theta_0_current- derivada_0*alpha
        theta_1_updated = theta_1_current- derivada_1*alpha
        theta_2_updated= theta_2_current- derivada_2*alpha
        return theta_0_updated, theta_1_updated, theta_2_updated
    if(atributos==5):
        soma_2=0
        soma_3=0
        soma_4=0
        soma_5=0
        for i in range(0,tam):
            x1_aux=(x1[i]-min_x1)/(max_x1-min_x1)
            x2_aux=(x2[i]-min_x2)/(max_x2-min_x2)
            x3_aux=(x3[i]-min_x3)/(max_x3-min_x3)
            x4_aux=(x4[i]-min_x4)/(max_x4-min_x4)
            x5_aux=(x5[i]-min_x5)/(max_x5-min_x5)
            h0=theta_0_current+ theta_1_current*x1_aux+theta_2_current*x2_aux+theta_3_current*x3_aux+theta_4_current*x4_aux+theta_5_current*x5_aux
            soma_0+= h0 - y[i]
            soma_1+= (h0 - y[i]) * x1_aux
            soma_2+= (h0 - y[i]) * x2_aux
            soma_3+= (h0 - y[i]) * x3_aux
            soma_4+= (h0 - y[i]) * x4_aux
            soma_5+= (h0 - y[i]) * x5_aux
        derivada_0=2* soma_0/tam
        derivada_1=2* soma_1/tam
        derivada_2=2* soma_2/tam
        derivada_3=2* soma_3/tam
        derivada_4=2* soma_4/tam
        derivada_5=2* soma_5/tam
        theta_0_updated = theta_0_current- derivada_0*alpha
        theta_1_updated = theta_1_current- derivada_1*alpha
        theta_2_updated= theta_2_current- derivada_2*alpha
        theta_3_updated= theta_3_current- derivada_3*alpha
        theta_4_updated= theta_4_current- derivada_4*alpha
        theta_5_updated= theta_5_current- derivada_5*alpha
        return theta_0_updated, theta_1_updated, theta_2_updated, theta_3_updated,theta_4_updated, theta_5_updated

def gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, starting_theta_3, starting_theta_4, starting_theta_5, learning_rate, num_iterations):
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
    if (atributos>=2):
        theta_2=starting_theta_2
    if (atributos>=5):
        theta_3=starting_theta_3
        theta_4=starting_theta_4
        theta_5=starting_theta_5
    
    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []
    
    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    theta_2_progress = []
    theta_3_progress = []
    theta_4_progress = []
    theta_5_progress = []
    
    if (atributos==1):
        # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
        for i in range(num_iterations):
            cost_graph.append(compute_cost(theta_0, theta_1, 0, 0, 0, 0, data))
            theta_0, theta_1= step_gradient(theta_0, theta_1,0,0,0,0, data, alpha=learning_rate)
        
        return [theta_0, theta_1, cost_graph]
    if (atributos==2):
        # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
        for i in range(num_iterations):
            cost_graph.append(compute_cost(theta_0, theta_1,theta_2,0,0,0,data))
            theta_0, theta_1,theta_2= step_gradient(theta_0, theta_1,theta_2,0,0,0,data, alpha=learning_rate)
        
        return [theta_0, theta_1,theta_2, cost_graph]
    if (atributos==5):
        # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
        for i in range(num_iterations):
            cost_graph.append(compute_cost(theta_0, theta_1,theta_2,theta_3,theta_4,theta_5, data))
            theta_0, theta_1,theta_2,theta_3,theta_4,theta_5 = step_gradient(theta_0, theta_1,theta_2,theta_3,theta_4,theta_5,data, alpha=learning_rate)
        
        return [theta_0, theta_1, theta_2,theta_3,theta_4,theta_5, cost_graph]
      
   
#Scripts

attr = sys.argv[1]
atributos = 0
data = np.genfromtxt(sys.argv[2], delimiter=',')

if attr == '1attr':
	atributos = 1
	theta_0, theta_1, cost_graph= gradient_descent(data, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, starting_theta_3=0, starting_theta_4=0, starting_theta_5=0, learning_rate=0.000001, num_iterations=int(sys.argv[3]))
elif attr == '2attr':
	atributos = 2
	theta_0, theta_1,theta_2, cost_graph, theta_0_progress= gradient_descent(data, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, starting_theta_3=0, starting_theta_4=0, starting_theta_5=0, learning_rate=0.000001, num_iterations=int(sys.argv[3]))
elif attr == '5attr':
	atributos = 5
	theta_0, theta_1, theta_2,theta_3,theta_4,theta_5, cost_graph = gradient_descent(data, starting_theta_0=0, starting_theta_1=0, starting_theta_2=0, starting_theta_3=0, starting_theta_4=0, starting_theta_5=0, learning_rate=0.000001, num_iterations=int(sys.argv[3]))


#Imprimir parâmetros otimizados
print ('Theta_0: ', theta_0)
print ('Theta_1: ', theta_1)
if(atributos>=2):
    print ('Theta_2: ', theta_2)
if(atributos>=5):
    print ('Theta_3: ', theta_3)
    print ('Theta_4: ', theta_4)
    print ('Theta_5: ', theta_5)

#Imprimir erro com os parâmetros otimizados
if (atributos==1):
    print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1,0,0,0,0, data))
if (atributos==2):
    print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1,theta_2,0,0,0, data))
if (atributos==5):
    print ('Erro quadratico medio: ', compute_cost(theta_0, theta_1, theta_2,theta_3,theta_4,theta_5, data))