
import pandas as pd
import numpy as np


# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous. (final_weight)
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# age, workclass, final_weight, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, class


def transform_col( data ):
    vlr_orig, values, count = np.unique(data, return_inverse=True, return_counts=True)
    result = {}
    result['vlr-orig'] = list(vlr_orig)
    result['values'] = list(values)
    result['vlr-count'] = list(count)
    return result



def data_set_v2(fname):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname, skipinitialspace=True, skip_blank_lines=True)
    cols = list(data.columns)
    process = ['workclass', 'education', 'occupation']
    for colname in process:
        if colname not in cols: continue
        dados = data[ colname ]
        ret = transform_col( dados )
        ret['colname'] = colname
        data.drop( columns=colname )
        data[ colname ] = ret['values']

    print(data)
    result['dados'] = data
    return result


def dataset_info(data):
    ###################
    data.info(verbose=True)
    print(data.describe())
    print('tipos:', data.dtypes)
    print('dimensoes:', data.ndim)
    print('linhas x colunas:', data.shape)
    ###################


def data_set( fname ):
    result = {}
    result['nome-arquivo'] = fname
    data = pd.read_csv(fname)

    cols = data.columns

    ultima = cols[-1]
    nome_orig = data[ultima]
    cls_orig, classes, cls_cnt = np.unique(nome_orig, return_inverse=True, return_counts=True)

    df = data.drop( columns=ultima )

    result['dados'] = df
    result['classes'] = classes
    result['cls-orig'] = cls_orig
    result['cls-count'] = cls_cnt

    return result


def show_dataset(data):
    print('-'*40)

    dataset_info(data)
    ncls = len( data['cls-orig'] )
    print(f'1- possui {ncls} classes')
    print('2- numero de itens para cada classe:', data['cls-count'])
    print('-'*40)



def show_unbalanced(data):
    soma = np.sum( data['cls-count'] )
    result = []
    for vlr in data['cls-count']:
        result.append( soma / vlr )
    max = np.max( result )
    print('desbalanceamento -->', max)


FNAME = 'adult.csv'

if __name__ == '__main__':
    data = data_set_v2(FNAME)
    
    #show_dataset(data)
    #show_unbalanced(data)

