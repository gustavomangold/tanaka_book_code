# Primeiro capítulo do livro

### Seção 1.3
import numpy as np
import matplotlib.pyplot as plt

# função de max likelihood, como definida no livro
# retorna um dicionário
def max_likelihood(n_observations, p: list, q: list):
	W = len(p)
	relative_entropy = 0
	for index in range(0, W):
		relative_entropy += p[index] * np.log(p[index]/q[index])
	probability = np.exp(-n_observations*relative_entropy)
	
	return {'Relative-Entropy': relative_entropy, 'Probability' : probability}

# fazemos um teste simples para verificar os valores da entropia relativa
# geramos 100 valores de acordo com a distribuiçao normal
# e comparamos com 100 valores com erro aleatorio
erro = 0.01
probability_exact = [np.random.uniform(0, 1) for x in range(0, 10)]
probability_guesses = [x + erro*np.random.uniform(0, 1) for x in probability_exact]

# segundo o livro, na prática não sabemos as probabilidades p_i, 
#então um modelo de machine learning serviria para chutar as probabilidades 
# q_i de modo que o erro (ou, nesse caso, a entropia relativa) seja minimizada

print('Entropia relativa: ', max_likelihood(10000, probability_exact, probability_guesses)['Relative-Entropy'])
