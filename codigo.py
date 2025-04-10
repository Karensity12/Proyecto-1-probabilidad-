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

def informacion(dat,categorico=False):
    lista = []    
    if categorico == False:
        # Intentamos tratar los datos como numéricos
        dat = pd.to_numeric(dat)
        lista.append(np.mean(dat))  # media
        lista.append(np.median(dat))  # mediana
        moda = dat.mode().iloc[0] if not dat.mode().empty else np.nan
        lista.append(moda)
        lista.append(np.std(dat))  # SD
        lista.append(np.mean(np.abs(dat - np.median(dat))))  # MAD
        lista.append(np.var(dat))  # Varianza
        Q1 = np.percentile(dat, 25)
        Q3 = np.percentile(dat, 75)
        lista.append(Q3 - Q1)  # IQR
        lista.append((np.std(dat, ddof=1)) / np.mean(dat) * 100)  # CV
        lista.append((np.mean(np.abs(dat - np.median(dat))) / np.median(dat) * 100) if np.median(dat) != 0 else np.nan)  # CVM
    elif categorico == True:
        lista = [np.nan] * 9  # media, mediana, moda, SD, MAD, varianza, IQR, CV, CVM
        lista[2] = dat.mode().iloc[0] if not dat.mode().empty else np.nan  # moda
    
    return lista

def atipicos (valor):    
    atipico = []
    valor = pd.to_numeric(valor)
    Q1 = np.nanpercentile(valor,25)    
    Q3 =np.nanpercentile(valor,75)
    iqr = Q3-Q1

    for x in valor:
        if x < (Q1 - 1.5*iqr) :
            atipico.append(x)
        elif x > (Q3 + 1.5*iqr):
            atipico.append(x)

    cont = len(atipico)
    return  cont,atipico

def mostrar_atipicos(lista : list, df):
    for i in range(len(lista)):
        cont,datos = atipicos(df[lista[i]])
        print(f"{lista[i]} tiene {cont} datos atipicos")
        print("Los datos atipicos son:")
        print(datos)

#%%
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

resultados = {'': ["Media", "Mediana", "Moda", "SD", "MAD", "Varianza", "IQR", "CV", "CVM"],    
    'aveOralF': informacion(csv_filter['aveOralF']),
    'aveOralM': informacion(csv_filter['aveOralM']),
    'Gender': informacion(csv_filter['Gender'], categorico=True),
    'Age': informacion(csv_filter['Age'], categorico=True),
    'Ethnicity': informacion(csv_filter['Ethnicity'], categorico=True), 
    'T_atm': informacion(csv_filter['T_atm']),
    'Humidity': informacion(csv_filter['Humidity']),
    'Cosmetics': informacion(csv_filter['Cosmetics'], categorico=True),
    'PromMax1R13': informacion(csv_filter['PromMax1R13'])
}

df = pd.DataFrame(resultados)
#%%
#Datos atipicos
lista = ["aveOralF",
         "aveOralM",  
         "T_atm", 
         "Humidity",
         "PromMax1R13"] 
   
mostrar_atipicos(lista, csv_filter)

# %%
