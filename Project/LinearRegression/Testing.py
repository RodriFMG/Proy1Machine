from Project.ExtractData.Extraer import ExtractAirQualityData
import numpy as np


def Testing(model):
    x_test, y_test = ExtractAirQualityData(Data="test", Normalization="Standard")

    ColumnOnes = np.full((x_test.shape[0], 1), 1)
    x_test = np.concatenate([x_test, ColumnOnes], axis=1)

    y_pred = model.Predict(x_test)

    # Accucary
    tolerancia = 0.1
    corrects = np.sum(np.abs(y_test - y_pred) <= tolerancia)
    acc = corrects / len(y_test)
    print(f"Precisión: {acc * 100:.5f}%")

    # MSE
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    print(f"MSE: {mse:.5f}")
    print(f"RMSE: {rmse:.5f}")


