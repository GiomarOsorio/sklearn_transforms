from sklearn.base import BaseEstimator, TransformerMixin


# All sklearn Transforms must have the `transform` and `fit` methods
class Custom_DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data.drop(labels=self.columns, axis='columns')

class Custom_CategoricalColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        data = X.copy()
        for column in self.columns:
            data[column] = data[column].astype('category').cat.codes
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return data
    
class Custom_Scaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, range_min=0, range_max=1):
        data = X.copy()
        groups = self.generate_groups(data)
        for group in groups:
            data.update(self.min_max_scaler(data[group], range_min, range_max))
        return data

    def generate_groups(self, X):
        columns_name = list(X.columns.values)
        groups = []
        group = [columns_name[0]]
        name_group_prefix = columns_name[0].split('_')[0]
        for index in range(1,len(columns_name)):
            column_name_prefix = columns_name[index].split('_')[0]
            if column_name_prefix == name_group_prefix:
                group.append(columns_name[index])
            else:
                groups.append(group)
                group = [columns_name[index]]
                name_group_prefix = column_name_prefix
        groups.append(group)
        return groups

    def min_max_scaler(self, X, range_min, range_max):
        data = X.copy()
        data_min = self.min_global(data)
        data_max = self.max_global(data)
        data = data.apply(lambda x: self.range_scaler(x, data_min, data_max, range_min, range_max))
        return data
    
    def min_global(self, X):
        data = X.copy()
        return data.min().min()

    def max_global(self, X):
        data = X.copy()
        return data.max().max()

    def range_scaler(self, X, data_min, data_max, range_min, range_max):
        data_std = (X - data_min) / (data_max - data_min)
        data_scale = data_std*(range_max - range_min) + range_min
        return data_scale
