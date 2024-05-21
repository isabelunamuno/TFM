#!/usr/bin/env python
# coding: utf-8

# ## ANÁLISIS DE DATOS

# Importar los paquetes necesarios

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import pyexcel as p 
import random
warnings.filterwarnings("ignore")
from matplotlib.dates import DateFormatter


# Cargar el conjunto de datos

# In[2]:


inventario_inicial=pd.read_csv('BegInvFINAL12312016.csv')
inventario_final=pd.read_csv('EndInvFINAL12312016.csv')
compras=pd.read_csv('PurchasesFINAL12312016.csv')
precio_compras=pd.read_csv('2017PurchasePricesDec.csv')
factura_compras=pd.read_csv('InvoicePurchases12312016.csv')
ventas=pd.read_csv('SalesFINAL12312016.csv')


# Para tener una idea general de los datos, se han impreso las primeros 5lineas de cada tabla

# In[3]:


inventario_inicial.head()


# In[4]:


inventario_final.head()


# In[5]:


compras.head()


# In[6]:


precio_compras.head()


# In[7]:


factura_compras.head()


# In[8]:


ventas.head()


# ### Análisis exploratorio de datos
# 
# A continuación, se ha realizado el análisis exploratorio de datos, obteniendo una tabla donde se tiene detalles clave sobre los conjuntos de datos, como numero de columnas y filas, tipos de datos, valores no informados... Esta visión global ayuda a identificar errores, contribuyendo a la posterior limpieza y preprocesamiento de los datos.

# In[9]:


# Se ha creado una función donde devuelve una tabla con la info necesaria para obtener información de los datos.
def dataframe_info(df):
    report = pd.DataFrame(columns=['Columna', 'TipoDato', 'Filas', 'Valores Unicos', 'Missings', 'Missings (%)'])
    for column in df.columns:
        data_type = df[column].dtype
        rows = df[column].shape[0] 
        unique_count = df[column].nunique()
        missing_values = df[column].isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        report.loc[len(report)] = [column, data_type, rows, unique_count,  missing_values, missing_percentage.round(4)]
    return report


# In[10]:


dataframe_info(inventario_inicial)


# In[11]:


dataframe_info(inventario_final)


# In[12]:


dataframe_info(compras)


# In[13]:


dataframe_info(precio_compras)


# In[14]:


dataframe_info(factura_compras)


# In[15]:


dataframe_info(ventas)


# La exploración de datos ayuda a identificar irregularidades que necesitan limpieza y preprocesamiento:
# 
# - Algunas tablas contienen missings que es necesario identificar y tratar adecuadamente.
# - Las columnas de tamaño en todas las tablas muestran incoherencias debido a la variación de las unidades de medida. Algunos ejemplos son 750mL, Litro, 750mL + 2/,2 Pk, 50mL 4 Pk, 750mL 3 Pk. Es necesario convertirlas a una unidad de medida única y coherente.
# - Las columnas Tamaño y Volumen llevan datos duplicados y una de ellas puede eliminar.
# - El formato de fecha difiere entre tablas. Por ejemplo, la tabla de ventas utiliza d/m/aaaa, mientras que otras utilizan aaaa/mm/dd.
# - Algunas columnas contienen espacios finales entre las entradas, como se ve en la columna VendorName de la tabla de compras (por ejemplo, ALTAMAR BRANDS LLC ,)

# ### Limpieza y preprocesamiento de datos
#  

# A. Tratamiento de Missings:
# 
# - Las tablas "inventario_inicial" y "ventas" no tienen valores faltantes, por lo que nos concentraremos en las otras cuatro tablas.
# 
# - (A.1) En la tabla "inventario_final" en la columna "city": De un total de 224.489 entradas, faltan 1.284 valores. Primero se ha observado si todos los nulos de ciudad pertenecen a la misma tienda, lo cual se ha visto que si.Por lo tanto, se le ha asignado el nombre de la ciudad que pertence a es tienda, obteniendo el datos de la tabla inventario inicial.
# 
# - (A.2) Missings en la tabla "compras" en la columna "size": Sólo faltan 3 valores de un total de 2.372.474 entradas, por lo que se ha decidiso eliminar esas filas. 
# 
# - (A.3) Missings en la tabla "Ventas" en la columna "Approval": Faltan informacion en 5.169 registros de un total de 5.543, lo que supone el 93% del total.  Dado el alto porcentaje de valores faltantes, se ha decidido eliminar la columna.
# 
# - (A.4) Missings en la tabla "precio_compras": Faltan valores individuales en las columnas "Description", "Size" y "Volume". En estos casos como estamos hablando de un solo registro en cada caso, se ha decidido elimar dichas filas.

# In[16]:


# A.1
# Filtrar las filas donde 'City' es NaN
city_nan = inventario_final[inventario_final['City'].isna()]

# Obtener los números de tienda de aquellas ciudades que tienen NaN en 'City'
tienda_city_nan = city_nan['Store'].unique()
print(f"Números de tienda con 'City' NaN: {tienda_city_nan}")

# Verificar si alguna vez la tienda 46 tiene un valor diferente de NaN en 'City'
store_46_rows = inventario_final[inventario_final['Store'] == 46]
store_46_non_nan_city = store_46_rows['City'].notna().any()


if store_46_non_nan_city:
    print("La tienda 46 tiene al menos un valor no NaN en la columna 'City'.")
else:
    print("Todos los valores de 'City' para la tienda 46 son NaN.")


# In[17]:


#Se ha comprobado cual es la tienda 46 en la tabla inventario incial, obteniendo el resultado TYWARDREATH
store_46_rows = inventario_inicial[inventario_inicial['Store'] == 46]
store_46_rows


# In[18]:


#Tratamiento de los Missing

#A.1
inventario_final["City"] = inventario_final["City"].fillna("TYWARDREATH")

#A.2
compras = compras[compras['Size'].notna()]
#purchases = compras[compras['Size'].dropna()]

#A.3 
factura_compras = factura_compras.drop(['Approval'], axis=1)

#A.4
precio_compras.dropna(subset=["Description"], inplace=True)
precio_compras.dropna(subset=["Size"], inplace=True)
precio_compras.dropna(subset=["Volume"], inplace=True)

# Funcion para confirmar que se han eliminado los missings
def confirmar_missings(*datasets):
    for df_name, df in datasets:
        contador_missings = {}
        for idx, column_name in enumerate(df.columns):
            missing_count =df[column_name].isnull().sum()
            if missing_count > 0:
                contador_missings[column_name] = missing_count
        
        if len(contador_missings) == 0:
            print(f"Valores no informados en {df_name} = {len(contador_missings)}")
        else:
            for column_name, count in contador_missings.items():
                print(f"Columna: {column_name}, Valores no informados: {count}")
       
confirmar_missings(("Inventario inicial", inventario_inicial), ("Inventario final", inventario_final), ("Compras", compras), ("Factura compras", factura_compras), ("Precio compras", precio_compras),("Ventas", ventas))


# B. Tratamiento de duplicados
# 
# A continuación se ha comprobado si las bases de datos contienen valores duplicados.

# In[19]:


# Comprobacion de duplicados
datasets = [inventario_inicial, inventario_final, compras, factura_compras, precio_compras, ventas]

for i, df in enumerate(datasets, start=1):
    duplicados = df.duplicated().any()
    if duplicados:
        print(f"El dataset {i} tiene duplicados.")
    else:
        print(f"El dataset {i} no tiene duplicados.")
    print("-" * 35)  # Separador entre datasets


# C. Irregularidades en las entradas de datos.
# 
# La columna Size varía en todos las tablas, excepto en el conjunto de datos precio_compras. La información sobre el tamaño, que representa el volumen, se introduce en varios formatos, como litro, mililitro, onza, paquete (pk) y combinaciones de los mismos. Por lo tanto, estas medidas se tienen que convertir en un litro estándar unificado. Primero de todo se han comprobado los recuentos únicos. A los que tengan un solo recuento y no presenten un patrón, se les a asignado una tasa equivalente. A los que tengan varios recuentos y un patrón, se han convertido según los patrones.

# In[20]:


# Se ha observado el patron, obteniendo un recuento unico de cada entrada de tamaño en todos os conjuntos de datos 
data_frames = [inventario_inicial, inventario_final, precio_compras, compras, ventas]
medidas = pd.concat([df["Size"] for df in data_frames])
medidas_unicas = medidas.unique()
size_counts = {}
for size in medidas_unicas:
    size_counts[size] = medidas[medidas == size].count()
medidas_unicas_count = pd.DataFrame({
    'Size': medidas_unicas,
    'Total Count': [size_counts[size] for size in medidas_unicas]})
print(medidas_unicas_count)


# A partir de la salida de recuentos únicos, se ha observado  que hay 51 formas diferentes de expresar medidas de volumen.
# Se han utilizado expresiones regulares para recorrerlos y aplicar tipos de conversión. Se han agrupado en tres patrones:
# 1) Ejemplos: 750ml, 750mL, 750ml + 3/, 750 4p, 750 4pk, etc.
# 2) Patrones como Litro, L o l
# 3) Tamaños expresados en Oz.
# 
# Se ha creado una funcion para aplicar estas conversiones a todos los conjuntos de datos.

# In[21]:


# Manejar las irregularidades de tamaño en todos los conjuntos de datos(datasets) 
def convert_to_liters(size):
    size = str(size).lower()
    # Convert 3/100ml and 5/2 -->  
    if "3/100ml" in size: return round(3 * 0.01, 2)  #3/100ml--> 3 packs of 100ml 
    elif '5/2 oz' in size: return round(10 * 0.0295735, 2)  #5/2 oz --> 5 packs of 1/2
    
    #pasar de galón a litro
    elif 'gal' in size:
        gal_value = float(re.search(r'\d+\.*\d*', size).group())
        return round(gal_value * 3.786, 2)   
    
    #patrones como 750 ml, 750 ml, 750 ml + 3/, 750 4p, 750 4pk, etc.
    elif 'ml' in size:
        if 'p' in size:
            ml_value, pack_value = re.search(r'(\d+\.*\d*)\s*m*l*\s*(\d*)\s*p*/*P*k*/*', size).groups()
            ml_value = float(ml_value)
            pack_value = float(pack_value) if pack_value else 1
            return round(ml_value * pack_value / 1000, 2)  # p or pk means pack--> multiply
        elif '+' in size:
            ml_value, pack_value = map(float, re.findall(r'\d+\.*\d*', size))
            return round((ml_value + (pack_value * 50)) / 1000, 2)  
        else:
            ml_value = float(re.search(r'\d+\.*\d*', size).group())
            return round(ml_value / 1000, 2)  
    
    # convertir Litro o L o l
    elif 'liter' in size or 'l' in size:
        if size == 'liter' or size == 'l': return 1.00
        else:
            liter_value = float(re.search(r'\d+\.*\d*', size).group())
            return round(liter_value, 2)  
    
    #onzas a litros    
    elif 'oz' in size:
        oz_value = float(re.search(r'\d+\.*\d*', size).group())
        return round(oz_value * 0.0295735, 2) 
    else:
        return None

datasets = [inventario_inicial, inventario_final, compras, precio_compras, ventas]

# Aplicar la función de conversión a cada dataset
for i, dataset in enumerate(datasets):
    dataset['Size'] = dataset['Size'].apply(convert_to_liters)
    datasets[i] = dataset


# In[22]:


# Eliminar los espacios finales en las columnas especificadas
def remove_spaces(df, *columns):
    for col in columns:
        df[col] = df[col].str.strip()
    return df
inventario_inicial = remove_spaces(inventario_inicial, "City", "Description")
inventario_final = remove_spaces(inventario_final, "City", "Description")
compras= remove_spaces(compras, "Description", "VendorName")
factura_compras = remove_spaces(factura_compras, "VendorName")
precio_compras = remove_spaces(precio_compras, "Description", "VendorName")
ventas = remove_spaces(ventas, "Description", "VendorName")

# Renombrar 'vendorNo' a 'VendorNumber' en las ventas para mantener uniformidad entre las diferentes tablas.
ventas.rename(columns={'VendorNo': 'VendorNumber'}, inplace=True) 


# Normalizar el formato de hora de la columna especificada en un DataFrame: Las columnas de fecha de los conjuntos de datos Inventario inicial e Inventario final ya tienen formato de fecha, por lo que no necesitamos formatearlas. Formatearemos el resto.

# In[23]:


def format_time(df, *columns):
    for col in columns:
        # Convertir las fechas a objetos datetime
        df[col] = pd.to_datetime(df[col], errors='coerce')
        # Formatear las fechas en el formato deseado
        df[col] = df[col].dt.strftime('%d/%m/%Y')
    return df

# Aplicar la función a los DataFrames correspondientes
inventario_inicial = format_time(inventario_inicial, "startDate")
inventario_final = format_time(inventario_final, "endDate")
compras = format_time(compras, "PODate", "ReceivingDate", "InvoiceDate", "PayDate")
factura_compras = format_time(factura_compras, "InvoiceDate", "PODate", "PayDate")
ventas = format_time(ventas, "SalesDate")


# Tambien se ha visto que los nombres de las ciudades no son reales, por lo que se han sustituido por aquellos que si lo son. 

# In[24]:


city_counts_begin = inventario_inicial['City'].value_counts()
city_counts_df = pd.DataFrame({'City': city_counts_begin.index, 'Count': city_counts_begin.values})
city_counts_df.to_excel('city_counts_begin.xlsx', index=False)


# In[25]:


city_counts_end = inventario_final['City'].value_counts()
city_counts_df = pd.DataFrame({'City': city_counts_end.index, 'Count': city_counts_end.values})
city_counts_df.to_excel('city_counts_end.xlsx', index=False)


# Se han pasado ambos excels y se han juntado todas las ciudades (las que se encuentran en las dos tablas) en una columna llamada "Invented_city"

# In[26]:


# Leer el archivo CSV con las ciudades inventadas y las ciudades reales
df_ciudades = pd.read_csv('Ciudades.csv', sep=';', encoding='latin1')
print(df_ciudades.head())


# In[27]:


# Reemplazar las ciudades inventadas con las ciudades reales
inventario_inicial['City'] = inventario_inicial['City'].replace(dict(zip(df_ciudades['Invented_city'], df_ciudades['Real_city'])))
inventario_final['City'] = inventario_final['City'].replace(dict(zip(df_ciudades['Invented_city'], df_ciudades['Real_city'])))


# Una vez realizado el cambio de ciudades, tambien se ha modificado la columna InventoryId, que procede de la columna ciudad.

# In[28]:


# Modificar la columna 'Inventory_Id' en función de los cambios en 'city' y 'brand'
inventario_inicial['Store'] = inventario_inicial['Store'].astype(str)
inventario_inicial['Brand'] = inventario_inicial['Brand'].astype(str)
inventario_inicial['InventoryId'] = inventario_inicial['Store'] + '_' + inventario_inicial['City'] + '_' + inventario_inicial['Brand']

inventario_final['Store'] = inventario_final['Store'].astype(str)
inventario_final['Brand'] = inventario_final['Brand'].astype(str)
inventario_final['InventoryId'] = inventario_final['Store'] + '_' + inventario_final['City'] + '_' + inventario_final['Brand']


# In[29]:


# Convertir 'Store' y 'Brand' a cadenas de caracteres
ventas['Store'] = ventas['Store'].astype(str)
ventas['Brand'] = ventas['Brand'].astype(str)

# Fusionar 'sales' y 'begin_inventory' en función de 'Store' y 'Brand'
merged_sales = pd.merge(ventas, inventario_inicial[['Store', 'Brand', 'InventoryId']], on=['Store', 'Brand'], how='left')

# Actualizar 'InventoryId' en 'sales' con los valores fusionados
ventas['InventoryId'] = merged_sales['InventoryId_y']


#Lo mismo para la tabla compras
compras['Store'] = compras['Store'].astype(str)
compras['Brand'] = compras['Brand'].astype(str)

# Fusionar 'sales' y 'begin_inventory' en función de 'Store' y 'Brand'
merged_sales = pd.merge(compras, inventario_inicial[['Store', 'Brand', 'InventoryId']], on=['Store', 'Brand'], how='left')

# Actualizar 'InventoryId' en 'sales' con los valores fusionados
compras['InventoryId'] = merged_sales['InventoryId_y']


# Comprobar que se ha realizado el cambio correctamente

# In[30]:


inventario_inicial.head()


# In[31]:


inventario_final.head()


# In[32]:


compras.head()


# In[33]:


ventas.head()


# ## Crear tablas maestras y tablas de hechos

# M_PRODUCTOS

# In[34]:


# Seleccionar las columnas deseadas
columnas_interesantes = ['Brand', 'Description','Size','Price']
M_productos = precio_compras.loc[:, columnas_interesantes]

#Cambiar el nombre de las columnas
M_productos.columns=['IdProducto', 'Producto','Volumen','Precio']

# Mostrar el nuevo DataFrame
M_productos


# In[35]:


# Encontrar filas duplicadas basadas en dos columnas
filas_duplicadas = M_productos[M_productos.duplicated(subset=['IdProducto', 'Producto'], keep=False)]
print("Número de filas duplicadas:", len(filas_duplicadas))


# M_CLIENTES

# In[36]:


print(precio_compras.columns)


# In[37]:


# Seleccionar las columnas deseadas
columnas_interesantes = ['VendorNumber', 'VendorName']
M_clientes = ventas.loc[:, columnas_interesantes]
#Eliminar duplicados
M_clientes.drop_duplicates(inplace=True)
#Cambiar el nombre de las columnas
M_clientes.columns=['IdCliente','Cliente']

# Mostrar el nuevo DataFrame
M_clientes


# M_PROVEEDORES

# In[38]:


# Selecciona las columnas deseadas
columnas_interesantes = ['VendorNumber', 'VendorName']
M_proveedores = precio_compras.loc[:, columnas_interesantes]

#Eliminar duplicados
M_proveedores.drop_duplicates(inplace=True)

#Cambiar el nombre de las columnas
M_proveedores.columns=['IdProveedor','Proveedor']

# Mostrar el nuevo DataFrame
M_proveedores


# M_TIENDAS

# In[39]:


# Selecciona las columnas deseadas
columnas_interesantes = ['Store', 'City']
M_tiendas = inventario_final.loc[:, columnas_interesantes]
#Eliminar duplicados
M_tiendas.drop_duplicates(inplace=True)
#Cambiar el nombre de las columnas
M_tiendas.columns=['IdTienda','Ciudad']

M_tiendas


# In[40]:


# Seleccionar las columnas deseadas
columnas_interesantes = ['VendorNumber', 'InvoiceDate','PONumber','PODate','PayDate']
M_facturas = factura_compras.loc[:, columnas_interesantes]
#Eliminar duplicados
M_facturas.drop_duplicates(inplace=True)
#Cambiar el nombre de las columnas
M_facturas.columns=['IdProveedor','FechaFactura','IdPuntoPedido','FechaPuntoPedido','FechaPago']

M_facturas


# H_VENTAS

# In[41]:


H_ventas = ventas[['Store','Brand','SalesQuantity','SalesDollars','SalesPrice','SalesDate','ExciseTax','VendorNumber']]
H_ventas.columns=['IdTienda','IdProducto','CantidadVentas','PrecioVenta','PrecioUnidad','FechaVenta','Impuesto','IdCliente']
H_ventas['PrecioTotal']= H_ventas['CantidadVentas'] * (H_ventas['PrecioVenta'] + H_ventas['Impuesto'])


# In[42]:


H_compras = compras[['Store','Brand','VendorNumber','PONumber','ReceivingDate','PODate','PurchasePrice','Quantity','Dollars']]
H_compras.columns=['IdTienda','IdProducto','IdProveedor','IdPuntoPedido','FechaLllegada','FechaPuntoPedido','PrecioCompraUnidad','Cantidad','PrecioCompraTotal']
H_compras


# In[43]:


H_begin_inventory = inventario_inicial[['InventoryId','Store','Brand','onHand','startDate']]
H_begin_inventory.columns=['IdInventario','IdTienda','IdProducto','Stock','Fecha']
H_begin_inventory


# In[44]:


H_end_inventory = inventario_final[['InventoryId','Store','Brand','onHand','endDate']]
H_end_inventory.columns=['IdInventario','IdTienda','IdProducto','Stock','Fecha']


# In[45]:


M_clientes.to_excel("M_clientes.xlsx", index=False, engine='xlsxwriter')
M_proveedores.to_excel("M_proveedores.xlsx", index=False, engine='xlsxwriter')
M_productos.to_excel("M_productos.xlsx", index=False, engine='xlsxwriter')
M_tiendas.to_excel("M_tiendas.xlsx", index=False, engine='xlsxwriter')
M_facturas.to_excel("M_facturas.xlsx", index=False, engine='xlsxwriter')


# In[46]:


H_ventas.to_excel("H_ventas.xlsx", index=False, engine='xlsxwriter')
#H_compras.to_excel("H_compras.xlsx", index=False, engine='xlsxwriter')
H_begin_inventory.to_excel("H_begin_inventory.xlsx", index=False, engine='xlsxwriter')
H_end_inventory.to_excel("H_end_inventory.xlsx", index=False, engine='xlsxwriter')


# Como el archivo de compras es demasiado grande se ha intentado leer con un interfaz mas sencilla.

# In[47]:


#Convertir el df a un diccionario de Python
data_dict = H_compras.to_dict(orient='records')

#Guardar el diccionario en un archivo excel
p.save_as(records=data_dict, dest_file_name="H_compras.xlsx")


# ABC-ANALYSIS

# In[54]:


# Fusionar los datos de las tablas de ventas y compras
Analisis_ABC = pd.merge(ventas, compras[['InventoryId', 'Store', 'Brand', 'Description', 
                                                   'Size', 'PurchasePrice']], 
                             on=['InventoryId', 'Store', 'Brand', 'Description', 'Size'])
# Seleccionar columnas relevantes
Analisis_ABC = Analisis_ABC[['Brand', 'Description', 'SalesQuantity', 'SalesPrice', 'PurchasePrice']]

# Agrupar por marca y descripción y calcular estadísticas
M_analisis_ABC = Analisis_ABC.groupby(['Brand', "Description"]).agg(
                                    CosteUnidad=("PurchasePrice", "mean"),
                                    VentaUnidad=("SalesPrice", "mean"),
                                    Demanda=('SalesQuantity', 'sum')).round(2).reset_index()

# Calcular Ventaspara todos los productos 
M_analisis_ABC["Ventas"] = ((M_analisis_ABC["VentaUnidad"] * M_analisis_ABC["Demanda"]) / 1e3).round(3)
M_analisis_ABC = M_analisis_ABC.sort_values(by=["Ventas"], ascending=False)

# Calcular la relación entre las Ventas de cada artículo y las Ventas totales
M_analisis_ABC["Ratio"] = (M_analisis_ABC["Ventas"].cumsum() / M_analisis_ABC["Ventas"].sum()).round(2)

# Asignar a cada elemento a la categoría definida 
def categorizar(ratio):
    if ratio < 0.7: return 'A'
    elif ratio < 0.9: return 'B'
    else: return 'C'

M_analisis_ABC["Categoria"] = M_analisis_ABC["Ratio"].apply(categorizar)
M_analisis_ABC = M_analisis_ABC.rename(columns={'Brand': 'IdProducto', 'Description': 'Producto'})

M_analisis_ABC


# In[55]:


M_analisis_ABC.to_excel("M_analisis_ABC.xlsx", index=False, engine='xlsxwriter')

