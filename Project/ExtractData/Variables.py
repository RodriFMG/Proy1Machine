PATH_DS = '../../DataSources/'

TypeData = {
    'test': PATH_DS + 'test_set_air_quality.csv',
    'train': PATH_DS + 'training_set_air_quality.csv',
    'validation': PATH_DS + 'validation_set_air_quality.csv'
}

# Si no se quiere considerar un atributo para el entrenamiento, solo quiten los campos.
AtributosInteres = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'TEMP', 'PRES',
                    'DEWP', 'RAIN', 'WSPM', 'target']

from .NormalizarData import StandardNormalization, MinMax

# Si se quiere agregar otro método de normalización, normal acá agregen a esta lista el nombre esperado
# para hacer más flexible el código :b.
TypeNormalization = {
    'MinMax': MinMax,
    'Standard': StandardNormalization
}