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

    def Predict(self, x):
        return np.dot(x, self.w)

    def LossFunction(self, y_pred):

        error = np.sum((self.y - y_pred) ** 2)
        return 1 / (2 * self.m) * error

    def GradientDescent(self, y_pred):
        error = self.y - y_pred
        return (-1 / self.m) * np.dot(self.x.T, error)

    def Backward(self, dw):
        self.w -= self.lr * dw

    def fit(self, MaxIters=1e+4, factor=1e+2, beta=0.9):

        CostHistory = []
        cost = 1
        NumIters = 0
        CostPonderado = 0

        while cost > self.umbral and NumIters < MaxIters:

            y_pred = self.forward()
            cost = self.LossFunction(y_pred)
            dw = self.GradientDescent(y_pred)
            self.Backward(dw)

            if NumIters > 1:
                CostPonderado = beta * CostPonderado + (1-beta) * cost
            else:
                CostPonderado = cost

            CostHistory.append(CostPonderado)
            CostHistory.append(cost)
            NumIters += 1

            if NumIters % factor == 0:
                print(f"Iteracion: {NumIters} --> Costo: {cost:.5f}")

        return CostHistory

    def GetCostHistory(self):

        if not self.CostHistory:
            print(f"Aún no se ha realizado el entrenamiento, o hubo un error.")
            return []

        return self.CostHistory