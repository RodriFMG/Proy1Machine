import pandas as pd
import numpy as np
from Variables import TypeData, AtributosInteres, TypeNormalization


# Data:
# train: datos de entrenamiento
# test: datos de testeo
# validation: datos de validación

# Normalization:
# Standard: Normalización tipo StandardScaler
# MinMax: Normalización basada en la relación de los datos con el mínimo y máximo del conjunto.
def ExtractAirQualityData(Data='train', Normalization='Standard'):
    if Data not in ['train', 'test', 'validation']:
        raise ValueError(f"Se esperaba el tipo de dato: train, test o validation, pero se mando: {Data}")

    if Normalization not in ['Standard', 'MinMax']:
        raise ValueError(f"Se esperaba alguna normalización implementada, pero se encontró: {Normalization}")

    np.random.seed(42)

    DataCSV = pd.read_csv(TypeData[Data])
    SizeData = len(DataCSV)

    TotalData = np.empty((SizeData, 0))

    for Atributo in AtributosInteres:
        AtributoData = DataCSV[Atributo].to_numpy()
        AtributoData = AtributoData.reshape(-1, 1)
        TotalData = np.concatenate([TotalData, AtributoData], axis=1)

    np.random.shuffle(TotalData)
    NormalizationFunction = TypeNormalization[Normalization]

    # Todos los atributos menos el último
    x = NormalizationFunction(TotalData[:, :-1])

    # Solo el último atributo
    y = TotalData[:, -1]

    return x, y
