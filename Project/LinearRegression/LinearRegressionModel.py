import numpy as np


# Regresión lineal usando MSE.
class LinearRegression:

    def __init__(self, lr, umbral, x, y):
        self.lr = lr
        self.umbral = umbral

        # 1, x.shape[1], para que sea en su formato matriz, ya que al concatenar
        # se debe presentar el mismo formato.
        ColumnOnes = np.full((x.shape[0], 1), 1)

        # x.shape -> # muestras, # dimensiones
        # concateno la fila de 1's, para las bayas.
        self.x = np.concatenate([x, ColumnOnes], axis=1)
        self.y = np.array(y)

        # se asigna un w para cada dimensión, incluyendo el b.
        self.w = np.random.rand(x.shape[1] + 1, 1)

        self.m = x.shape[0]

        self.CostHistory = self.fit()

    def forward(self):
        return np.dot(self.x, self.w)

    def LossFunction(self, y_pred):
        return 1 / (2 * self.m) * np.sum((self.y - y_pred) ** 2)

    def GradiantDescent(self, y_pred):
        return 1 / self.m * np.sum((self.y - y_pred) * self.x)

    def Backward(self, lr, dw):
        self.w -= lr * dw

    def fit(self, MaxIters=1e+5, factor=1e+2):

        CostHistory = []
        cost = 1
        NumIters = 0

        while cost > self.umbral and NumIters < MaxIters:

            y_pred = self.forward()
            cost = self.LossFunction(y_pred)
            dw = self.GradiantDescent(y_pred)
            self.Backward(self.lr, dw)

            CostHistory.append(cost)
            NumIters += 1

            if NumIters % factor == 0:
                print(f"Iteracion: {NumIters} --> Costo: {cost}")

        return CostHistory

    def GetCostHistory(self):
        return self.CostHistory
