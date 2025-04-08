import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

#Inspección de la base de datos 
aqui = {'Max1R13_1':3,'Max1R13_2':31,'Max1R13_3':60,'Max1R13_4':88,'aveOralF':114 ,'aveOralM':115,'Gender':116, 'Age':117, 'Ethnicity':118,'T_atm':119 , 'Humidity':120,'Cosmetics':122}
csv = pd.read_csv(r'FLIR_groups1and2.csv', delimiter=';')
csv = csv.iloc[2:].reset_index(drop=True) #Eliminación de las dos primeras filas
csv['PromMax1R13'] = csv[[csv.columns[aqui['Max1R13_1']], csv.columns[aqui['Max1R13_2']],csv.columns[aqui['Max1R13_3']],csv.columns[aqui['Max1R13_4']]]].astype(float).mean(axis=1)
csv_filter = csv.iloc[:, [aqui['aveOralF'], aqui['aveOralM'], aqui['Gender'], aqui['Age'], aqui['Ethnicity'], aqui['T_atm'], aqui['Humidity'], aqui['Cosmetics']]]
csv_filter.columns = ['aveOralF','aveOralM', 'Gender', 'Age', 'Ethnicity', 'T_atm', 'Humidity', 'Cosmetics']
csv_filter.loc[:,'PromMax1R13'] = csv['PromMax1R13']
#df_filled = csv_filter.fillna('nan')


#csv_filter = csv_filter.replace(' ', np.non)
csv_filter.loc[:,"Cosmetics"] = csv_filter["Cosmetics"].where(pd.notna(csv_filter["Cosmetics"]), None)
csv_filter.to_csv('FLIR_groups1and2_replace.csv', index=False)

#Funcion para calcular estadisticas
def informacion(dat):
    dat = dat.dropna()
    lista = []
    try:
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



#df = pd.DataFrame(resultados)
#print(df)


# analisis_aveOralF = informacion(csv_filter['aveOralF'])
# analisis_aveOralM = informacion(csv_filter['aveOralM'])
# analisis_Gender = informacion(csv_filter['Gender'])
# analisis_Age = informacion(csv_filter['Age'])
# analisis_Ethnicity = informacion(csv_filter['Ethnicity'])
# analisis_T_atm = informacion(csv_filter['T_atm'])
# analisis_Humidity = informacion(csv_filter['Humidity'])
# analisis_Cosmetics = informacion(csv_filter['Cosmetics'])

# df = pd.DataFrame({
#     'Media': analisis_aveOralF[0],
#     'Mediana': analisis_aveOralF[1],
#     'Moda': analisis_aveOralF[2],
#     'SD': analisis_aveOralF[3],
#     'MAD': analisis_aveOralF[4],
#     'Varianza': analisis_aveOralF[5],
#     'IQR': analisis_aveOralF[6],
#     'CV': analisis_aveOralF[7],
#     'CVM': analisis_aveOralF[8]
# })







# csv_filter = csv_filter.replace(' ', np.nan)
#csv_filter = csv_filter.dropna()
#seleccion = seleccion.reset_index(drop=True)



