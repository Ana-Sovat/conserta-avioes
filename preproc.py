#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:21:24 2025

@author: carolina
"""

print("Importando Módulos...")

import sys
import re

__import__('pysqlite3')

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import pandas as pd

dados = pd.read_csv('aircraft-historical-maintenance-dataset/Aircraft_Annotation_DataFile.csv')

# Abreviações

print("Tratando abreviações...")

abreviacoes = pd.read_csv("aircraft-historical-maintenance-dataset/Aviation_Abbreviation_Dataset.csv")

abreviacoes_sem_co = abreviacoes.loc[abreviacoes['Abbreviated']!='co']

dici_abrev = {linha[1]['Abbreviated']:linha[1]['Standard_Description'].upper() 
              for linha in abreviacoes_sem_co.iterrows()}

dici_abrev["L/H"] = "LEFT"
dici_abrev["R/H"] = "RIGHT"
dici_abrev['W/'] = "WITH"
dici_abrev['@'] = "AT"

dados['DESAB'] = dados['PROBLEM']

for abr in dici_abrev:
    dados['DESAB'] = dados['DESAB'].apply(lambda x:re.sub(fr"(?i)\b{abr}\b",dici_abrev[abr],x))

def abrev_co(string):
    if re.search(r"(?i)\bco\b",string):
        return True
    return False

mask_co = dados['DESAB'].apply(abrev_co)
frases_co = dados.loc[mask_co,'DESAB']

co_cutoff = [1556,1557,2164]
co_carbon = [5802,5803]

dados.iloc[co_cutoff,3] = dados.iloc[co_cutoff,3].apply(lambda x:re.sub(r"(?i)\bco\b","CUTOFF",x))
dados.iloc[co_carbon,3] = dados.iloc[co_carbon,3].apply(lambda x:re.sub(r"(?i)\bco\b","CARBON MONOXIDE",x))

# Grupos

for i in range(dados.shape[0]-1):
    if dados.iloc[i+1,3] == dados.iloc[i,3]:
        dados.iloc[i+1,0] = dados.iloc[i,0]

dados_agrupados = dados.groupby('IDENT').first()['DESAB']
ids = [str(a) for a in dados_agrupados.index.to_list()]
documentos = dados_agrupados.values.tolist()

dados.to_csv("dados_tratados.csv",index=False)
dados_agrupados.to_csv("problemas.csv")
