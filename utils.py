import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

class MinMax(BaseEstimator, TransformerMixin):
    def __init__(self, min_max_scaler=['idade', 'altura', 'peso']):
        self.min_max_scaler = min_max_scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if set(self.min_max_scaler).issubset(X.columns):
            min_max_enc = MinMaxScaler()
            X[self.min_max_scaler] = min_max_enc.fit_transform(X[self.min_max_scaler])
            return X
        else:
            print('Uma ou mais features não estão no DataFrame')
            return X
        
class OneHotEncodingNames(BaseEstimator, TransformerMixin):
    def __init__(self, OneHotEncoding=['meio_transporte']):
        self.OneHotEncoding = OneHotEncoding

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if set(self.OneHotEncoding).issubset(X.columns):

            # Aplicando o OneHotEncoder
            one_hot_enc = OneHotEncoder(sparse=False)
            one_hot_encoded_array = one_hot_enc.fit_transform(X[self.OneHotEncoding])
            feature_names = one_hot_enc.get_feature_names_out(self.OneHotEncoding)

            # Criando novo DataFrame com as colunas codificadas
            df_OneHotEncoding = pd.DataFrame(one_hot_encoded_array, columns=feature_names, index=X.index)

            # Mantendo o restante das colunas
            outras_features = [col for col in X.columns if col not in self.OneHotEncoding]
            df_concat = pd.concat([df_OneHotEncoding, X[outras_features]], axis=1)

            return df_concat

        else:
            print('Uma ou mais features não estão no DataFrame')
            return X
        
class OrdinalFeature(BaseEstimator, TransformerMixin):
    def __init__(self, ordinal_feature=['come_entre_refeicoes', 'freq_bebida_alcoolica']):
        self.ordinal_feature = ordinal_feature

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if set(self.ordinal_feature).issubset(X.columns):
            ordinal_encoder = OrdinalEncoder()
            X[self.ordinal_feature] = ordinal_encoder.fit_transform(X[self.ordinal_feature])
            return X
        else:
            print(f"Uma ou mais colunas {self.ordinal_feature} não estão no DataFrame")
            return X