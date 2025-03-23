import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


aqui = {'aveOralF':114 ,'aveOralM':115,'Gender':116, 'Age':117, 'Ethnicity':118,'T_atm':119 , 'Humidity':120,'Cosmetics':122}
csv = pd.read_csv(r'FLIR_groups1and2.csv', delimiter=';')
csv = csv.iloc[2:].reset_index(drop=True)
csv_filter = csv.iloc[:, [aqui['aveOralF'], aqui['aveOralM'], aqui['Gender'], aqui['Age'], aqui['Ethnicity'], aqui['T_atm'], aqui['Humidity'], aqui['Cosmetics']]]
csv_filter.columns = ["aveOralF","aveOralM", "Gender", "Age", "Ethnicity", "T_atm", "Humidity", "Cosmetics"]
#df_filled = csv_filter.fillna("nan")

#csv_filter = csv_filter.replace(' ', np.non)
csv_filter["Cosmetics"] = csv_filter["Cosmetics"].where(pd.notna(csv_filter["Cosmetics"]), None)
csv_filter.to_csv('FLIR_groups1and2_replace.csv', index=False)
#print(type(csv_filter['Cosmetics'][0]))

def informacion(dat):
    dat = dat.dropna()
    lista = []
    lista.append(np.mean(dat)) #media
    lista.append(np.median(dat)) #mediana
    moda_result = stats.mode(dat)  # Moda
    lista.append(moda_result.mode[0] if len(moda_result.mode) > 0 else np.nan)  # Moda
    lista.append(np.std(dat)) #SD
    lista.append(np.mean(np.abs(dat-np.median(dat)))) #MAD
    lista.append(np.var(dat)) #Varianza
    Q1 = np.nanpercentile(dat, 25)
    Q3 = np.nanpercentile(dat, 75)
    lista.append(Q3-Q1) #IQR
    lista.append(((np.std(dat,ddof = 1))/np.mean(dat)*100)) #CV
    lista.append((np.mean(np.abs(dat-np.median(dat)))/np.median(dat)*100)) #CVM
    return lista

resultados = {
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
print(df)

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



