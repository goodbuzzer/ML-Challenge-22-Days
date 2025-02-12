import numpy as np
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        # Taux d'apprentissage pour la descente de gradient
        self.lr = 0.001
        # Poids du modèle initialisés à None
        self.weights = None
        # Biais du modèle initialisé à None
        self.bias = None
        # Nombre d'itérations pour la descente de gradient
        self.n_iter = 1000
        
    def fit(self, X, y):
        # Nombre de lignes (échantillons) et de colonnes (caractéristiques) dans X
        n_rows, n_features = X.shape
        
        # Initialisation des poids à un vecteur de zéros de taille n_features
        self.weights = np.zeros(n_features)
        # Initialisation du biais à 0
        self.bias = 0
        
        # Boucle pour effectuer la descente de gradient n_iter fois
        for _ in range(self.n_iter):
            # Prédiction des valeurs de y en utilisant les poids et le biais actuels
            y_pred = np.dot(X, self.weights) + self.bias
        
            # Calcul du gradient pour les poids obtenu par derivation du gradient en fonction du poids w
            dw = (1 / n_rows) * np.dot(X.T, (y_pred - y))
            # Calcul du gradient pour le biais idem sur le biais
            db = (1 / n_rows) * np.sum(y_pred - y)
            
            # Mise à jour des poids en soustrayant le produit du taux d'apprentissage et du gradient
            self.weights -= self.lr * dw
            # Mise à jour du biais en soustrayant le produit du taux d'apprentissage et du gradient
            self.bias -= self.lr * db
    
    def predict(self, X):
        # Prédiction des valeurs de y en utilisant les poids et le biais appris
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
X = np.array([[1,2],[3,4],[5,6]]) 
y = np.array([2,4,6])

model = Model()

model.fit(X, y)

y_pred = model.predict(X)
print(y_pred)

# Tracer les résultats
plt.scatter(X[:, 0], y, color='blue', label='Données réelles')
plt.plot(X[:, 0], y_pred, color='red', label='Prédictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.savefig("regression_from_scratch.png")
plt.show()