import numpy as np


# Creo q las normalizaciones en regresiones se aplican solamente a los x, no se si se aplican al y xd.

# Normalización para que la mayor parte de los datos esten en un rango de [-3, 3], con
# mean aprox 0 y std aprox 1
def StandardNormalization(x_data):
    x_mean = np.mean(x_data)
    x_std = np.std(x_data)

    return (x_data - x_mean) / x_std


# Para que todos los datos estén en un rango de [0, 1]
def MinMax(x_data):
    x_min = np.min(x_data)
    x_max = np.max(x_data)

    return (x_data - x_min) / (x_max - x_min)
