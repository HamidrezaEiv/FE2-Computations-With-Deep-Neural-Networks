import numpy as np
import pandas as pd

class prep():
    def __init__(self, fdir):
        super(prep, self).__init__()
        self.fdir = fdir
        
    def train_test_sp(self, df):
        df = df.sample(frac=1, random_state=24).reset_index(drop=True)
        df_test = df.sample(frac=0.2, random_state=24)
        df_train = df.drop(df_test.index)
        return (df_train, df_test)
        
    def load(self):
        df = pd.read_csv(self.fdir, index_col=False)
            
        df = df.loc[:, (df != 0).any(axis=0)]
        self.keys = df.keys()
        return df
    
    def scale(self, scaler):
        df_train, df_test = self.train_test_sp(self.load())
        train = scaler(df_train)
        test = self.scale_df(df_test)
        return (train, test)
    
    def normscaler(self, df):
        X = df.values[:, :3]
        y = df.values[:, 3:6]
        y_x = df.values[:, 6:]
        
        a = np.std(X, axis = 0)
        b = np.std(y, axis = 0)
        c = 1 / np.concatenate([a / bi for bi in b])
        
        ma = np.mean(X, axis = 0)
        mb = np.mean(y, axis = 0)
        mc = 0.0
        
        X = (X - ma) / a     
        y = (y - mb) / b
        y_x = (y_x - mc) / c
        
        self.params = [a, b, c, ma, mb, mc]
        
        return (X, y, y_x)
    
    def scale_r(self, data):
        y = data[1]
        y_x = data[2]
        X = data[0] * self.params[0] + self.params[3]
        y = y * self.params[1] + self.params[4]
        y_x = y_x * self.params[2] + self.params[5]
        return (X, y, y_x)
    
    def scale_df(self, df):
        X = df.values[:, :3]
        y = df.values[:, 3:6]
        y_x = df.values[:, 6:]
        
        X = (X - self.params[3]) / self.params[0]    
        y = (y - self.params[4]) / self.params[1]
        y_x = (y_x - self.params[5]) / self.params[2]
        
        return (X, y, y_x)
    
    def scale_data(self, data):
        X = data[0]
        y = data[1]
        y_x = data[2]
        
        X = (X - self.params[3]) / self.params[0]    
        y = (y - self.params[4]) / self.params[1]
        y_x = (y_x - self.params[5]) / self.params[2]
        
        return (X, y, y_x)
    
    def data(self, df):
        X = df.values[:, :3]
        y = df.values[:, 3:6]
        y_x = df.values[:, 6:]
        
        return (X, y, y_x)
    
    def scale_x(self, x):
        return (x - self.params[3]) / self.params[0] 

