
import pandas as pd
import numpy as np


def _transform_col( data ):
    vlr_orig, values, count = np.unique(data, return_inverse=True, return_counts=True)
    result = {}
    result['vlr-orig'] = list(vlr_orig)
    result['values'] = list(values)
    result['vlr-count'] = list(count)
    return result


def _transform_data(data, col_list):

    for colname in list(data.columns):
        if colname not in col_list: continue
        dados = data[ colname ]
        ret = _transform_col( dados )
        ret['colname'] = colname
        data.drop( columns=colname )
        data[ colname ] = ret['values']

    return data



def dataset_info(data):
    ###################
    data.info(verbose=True)
    print(data.describe())
    print('tipos:', data.dtypes)
    print('dimensoes:', data.ndim)
    print('linhas x colunas:', data.shape)
    ###################

def remover_dados_faltantes( df ):
    mascara = df.apply(lambda linha: linha.astype(str).str.contains(r'\?')).any(axis=1)

    # Retorna um DataFrame apenas com as linhas que **não** contêm '?'
    data = df[~mascara].copy()
    return data


def data_set( fname ):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname, skipinitialspace=True, skip_blank_lines=True)
    dataset_info(data)

    data = remover_dados_faltantes( data )
    # data.to_csv('adult--removido.csv', index=False)
    dataset_info(data)

    mystr = 'workclass, education,  marital-status, occupation,     relationship,   race,   sex,    native-country, class'
    process = [x.strip() for x in mystr.split(',')]
    data = _transform_data(data, process)

    ultima = data.columns[-1]
    classes = list(data[ultima])
    df = data.drop( columns=ultima )

    result['dados'] = df
    result['classes'] = classes

    return result



