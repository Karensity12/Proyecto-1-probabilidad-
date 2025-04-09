import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

def convertir_NAN_MEAN(nombres,df):
    """Convierte los valores NaN de una columna a su media."""
    for i in nombres:
        df[i] = pd.to_numeric(df[i], errors='coerce')  # Convertir a numérico, forzando NaN donde no se puede
        df[i].fillna(np.mean(df[i]), inplace=True)

aqui = {'aveOralF':114 ,
'aveOralM':115,
'Gender':116, 
'Age':117, 
'Ethnicity':118,
'T_atm':119 , 
'Humidity':120,
'Cosmetics':122, 
'Max1R13_1':3,
'Max1R13_2':31,
'Max1R13_3':59,
'Max1R13_4':87}

csv = pd.read_csv(r'FLIR_groups1and2.csv', delimiter=';')
csv = csv.iloc[2:].reset_index(drop=True)


csv_filter = csv.iloc[:, [aqui['aveOralF'], aqui['aveOralM'], aqui['Gender'], aqui['Age'], aqui['Ethnicity'], aqui['T_atm'], aqui['Humidity'], aqui['Cosmetics']]]
csv_filter.columns = ["aveOralF","aveOralM", "Gender", "Age", "Ethnicity", "T_atm", "Humidity", "Cosmetics"]

data_temp = csv.iloc[:, [aqui['Max1R13_1'], aqui['Max1R13_2'], aqui['Max1R13_3'], aqui['Max1R13_4']]].astype(float)
data_temp.columns = ["Max1R13_1","Max1R13_2","Max1R13_3","Max1R13_4"]
convertir_NAN_MEAN(data_temp.columns,data_temp) 

csv_filter['PromMax1R13'] = data_temp.mean(axis=1)

csv_filter["Cosmetics"] = csv_filter["Cosmetics"].fillna('Nan')

csv_filter.to_csv('FLIR_groups1and2_replace.csv',  sep=';', decimal=',',index=False)

#%%
#Media, mediana, moda, SD, MAD, varianza, IQR, CV, CVM para aveOralF


#%%
#Funcion para calcular estadisticas
def informacion(dat,categorico=False):
    lista = []
    try:
        if categorico ==False:
            # Intentamos tratar los datos como numéricos
            dat = pd.to_numeric(dat)
            lista.append(np.mean(dat))  # media
            lista.append(np.median(dat))  # mediana
            moda = dat.mode().iloc[0] if not dat.mode().empty else np.nan
            lista.append(moda)
            lista.append(np.std(dat))  # SD
            lista.append(np.mean(np.abs(dat - np.median(dat))))  # MAD
            lista.append(np.var(dat))  # Varianza
            Q1 = np.nanpercentile(dat, 25)
            Q3 = np.nanpercentile(dat, 75)
            lista.append(Q3 - Q1)  # IQR
            lista.append((np.std(dat, ddof=1)) / np.mean(dat) * 100)  # CV
            lista.append(np.mean(np.abs(dat - np.median(dat))) / np.median(dat) * 100)  # CVM
        elif categorico == True:
            print
    except:
        # Si falla, es porque son datos categóricos
        for _ in range(2): lista.append(np.nan)  # media, mediana
        moda = dat.mode().iloc[0] if not dat.mode().empty else np.nan
        lista.append(moda)  # moda
        for _ in range(6): lista.append(np.nan)  # el resto no aplica

    return lista


df_modificado = csv_filter.copy()
ctg_vars = ['Gender', 'Ethnicity', 'Cosmetics','Age']
for col in ctg_vars:
    moda = df_modificado[col].mode()[0]  # Encuentra el valor más común
    df_modificado[col] = df_modificado[col].apply(lambda x: x if x == moda else np.nan)

# Convertir a numéricas solo las columnas que deben serlo
columnas_numericas = ['aveOralF', 'aveOralM', 'T_atm', 'Humidity', 'PromMax1R13']

for col in columnas_numericas:
    df_modificado.loc[:, col] = pd.to_numeric(df_modificado[col], errors='coerce')


resultados = {
    'aveOralF': informacion(df_modificado['aveOralF']),
    'aveOralM': informacion(df_modificado['aveOralM']),
    'Gender': informacion(df_modificado['Gender']),
    'Ethnicity': informacion(df_modificado['Ethnicity']),
    'T_atm': informacion(df_modificado['T_atm']),
    'Humidity': informacion(df_modificado['Humidity']),
    'Cosmetics': informacion(df_modificado['Cosmetics']),
    'PromMax1R13': informacion(df_modificado['PromMax1R13'])
}

estadisticas = ['Media', 'Mediana', 'Moda', 'SD', 'MAD', 'Varianza', 'IQR', 'CV (%)', 'CVM (%)']
df_estadisticas = pd.DataFrame(resultados, index= estadisticas)
print("\nResumen estadístico:")
print(df_estadisticas)


'''resultados = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralF': informacion(csv_filter['aveOralF']),
    'aveOralM': informacion(csv_filter['aveOralM']),
    'Gender': informacion(csv_filter['Gender']),
    'Age': informacion(csv_filter['Age']),
    'Ethnicity': informacion(csv_filter['Ethnicity']),
    'T_atm': informacion(csv_filter['T_atm']),
    'Humidity': informacion(csv_filter['Humidity']),
    'Cosmetics': informacion(csv_filter['Cosmetics'])
}

df = pd.DataFrame(resultados)
print(df)'''

