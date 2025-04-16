from LinearRegressionModel import LinearRegression
from Project.ExtractData.Extraer import ExtractAirQualityData
from Testing import Testing
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x_train, y_train = ExtractAirQualityData(Data="train", Normalization="Standard")

    lr = 0.005
    umbral = 0.005

    model = LinearRegression(lr, umbral, x_train, y_train)

    CostHistory = model.GetCostHistory()

    Testing(model=model)

    # Graficar la función de costo para la regresión lineal
    plt.plot(range(len(CostHistory)), CostHistory)
    plt.xlabel('Iteraciones')
    plt.ylabel('Función de Costo')
    plt.title('Regresión Lineal')

    plt.show()
