import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from keras.models import load_model
import os
import random
import get_model
from sklearn.model_selection import StratifiedKFold
import string
import keras
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import RocAucEvaluation as AUC
import DataGenerator as dg
import get_modelv2_3
from keras import backend as K
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'  
from keras.utils import plot_model  
import sys
from sklearn.metrics import roc_curve, auc 
from keras.backend.tensorflow_backend import set_session
import re
seed_value= 2020
random.seed(seed_value)
np.random.seed(seed_value)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# def focal_loss(gamma=2., alpha=.25):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

class Regression_solve():
    def __init__(self,n=8,batch_size=128,epochs=300,lr=0.0001,patience=10,alpha=0.5):#lr=0.0007
        self.n=n
        self.batch_size = batch_size  
        self.epochs =epochs
        self.lr = lr 
        self.patience = patience
        self.alpha=alpha

        self.font1 = {
            'weight': 'normal',
            'size': 16,
        }
        self.font2 = {
            'weight': 'normal',
            'size': 23,
        }

    def mean_squared_error_l2(self, y_true, y_pred, lmbda=0.01):
        cost = K.mean(K.square(y_pred - y_true))
        weights = []
        for layer in self.model.layers:
            weights = weights + layer.get_weights()
        result = tf.reduce_sum([tf.reduce_sum(tf.pow(wi, 2)) for wi in weights])
        l2 = lmbda * result 
        return cost + l2

    def predict_result(self,x_prot,x_comp,y_value,type):
        predict_result = self.model.predict([x_prot, x_comp])
        real = y_value
        df = pd.DataFrame(predict_result, columns=['predicted'])
        df['real'] = real
        df['set'] = type
        return df

    def save_predict_result(self,x_prot,x_comp,y_value,model_name,type, save_dir):
        df=self.predict_result(x_prot,x_comp,y_value,type)
        predict_file = os.path.join(save_dir,'regression_model_%s_%s_predict_result.csv' % (model_name,type))
        df.to_csv(predict_file, index=False)

    def computer_parameter(self, df,type):
        rmse = ((df['predicted'] - df['real']) ** 2).mean() ** 0.5
        mae = (np.abs(df['predicted'] - df['real'])).mean()
        corr = scipy.stats.pearsonr(df['predicted'], df['real'])
        lr = LinearRegression()
        lr.fit(df[['predicted']], df['real'])
        y_ = lr.predict(df[['predicted']])
        sd = (((df["real"] - y_) ** 2).sum() / (len(df) - 1)) ** 0.5
        print("%10s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f" % (type, rmse, mae, *corr, sd))
        with open('./suit_inception_v1_model_sel.txt', 'a') as fi:
            print("%10s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f" % (type, rmse, mae, *corr, sd), file=fi)
        return type, rmse, mae, corr, sd

    def computer_parameter_draw_scatter_plot(self,x_prot, x_comp, y_value,model_name,type, save_dir):
        sns.set(context='paper', style='white')
        sns.set_color_codes()
        df = self.predict_result(x_prot, x_comp, y_value, type)
        if all(df['real']>0):
            xlimb_start=0
        else:
            xlimb_start=-10
        if all(df['predicted']>0):
            ylimb_start=0
        else:
            ylimb_start=-10
        set_colors = {'train': 'b', 'validation': 'green', 'test': 'purple'}
        grid = sns.jointplot('real', 'predicted', data=df, stat_func=None, color=set_colors[type],
                             space=0.0, size=4, ratio=4, s=20, edgecolor='w', ylim=(ylimb_start, 16),
                             xlim=(xlimb_start, 16)
                             )
        grid.ax_joint.set_xticks(range(xlimb_start, 16, 5))
        grid.ax_joint.set_yticks(range(ylimb_start, 16, 5))  
        type, rmse, mae, corr, sd=self.computer_parameter(df,type)
        grid.ax_joint.text(1, 14, type + ' set', fontsize=14)  
        grid.ax_joint.text(16, 19.5, 'RMSE: %.2f ' % rmse)
        grid.ax_joint.text(16, 18.5, '(p): %.3f ' % corr[1])
        grid.ax_joint.text(16, 17.5, 'R2: %.2f ' % corr[0])
        grid.ax_joint.text(16, 16.5, 'SD: %.2f ' % sd)
        scatter_file = os.path.join(save_dir, '%s_%s_scatter_plot.png' %(model_name,type))
        grid.fig.savefig(scatter_file, dpi=400)

    def draw_loss_change(self,history,model_name, save_dir):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss_values, 'b', label='Training loss')
        plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
        plt.title('Training and validation loss', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        loss_file = os.path.join(save_dir,'%s_regression_training_validation_loss.png'%model_name)
        plt.savefig(loss_file)
        # mse
        plt.figure(figsize=(10, 10))
        plt.plot(epochs,history_dict['mse'], 'b',
                 label='Training mse')
        plt.plot(epochs,history_dict['val_mse'], 'r',
                 label='Validation mse')
        plt.tick_params(labelsize=20)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Mse', self.font2)
        plt.legend(prop=self.font1)
        mse_file = os.path.join(save_dir,'%s_regression_training_validation_mse.png'%model_name)
        plt.savefig(mse_file)

    def train_model(self, train_file, validation_file,  model_name):
        save_dir = os.path.join(os.getcwd(), 'models', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = get_model.get_model_regression(save_dir,self.alpha,                 
                    pro_branch_switch1 = 'inception_block', pro_branch_switch2 = 'inception_block',
                    pro_branch_switch3='inception_block_b', pro_add_attention = False,
                    comp_branch_switch1 = 'inception_block', comp_branch_switch2 = 'inception_block',
                    comp_branch_switch3 = 'inception_block_b', comp_add_attention = False)
        self.validation_x_prot, self.validation_x_comp, self.validation_y=dg.read_reg(validation_file)

        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(loss=self.mean_squared_error_l2, optimizer=optimizer,metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        bestfile = save_dir + "/%s_best_model.hdf5" % model_name
        checkpoint = ModelCheckpoint(bestfile, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')    
        # AUC
        RocAuc = AUC.RocAucEvaluation(validation_data=([self.validation_x_prot, self.validation_x_comp],
                                                       self.validation_y), interval=1)

        history = self.model.fit_generator(dg.read_reg_generator(train_file, self.batch_size),
                            steps_per_epoch=57,
                            epochs=self.epochs,
                            validation_data=([self.validation_x_prot, self.validation_x_comp], self.validation_y),
                            callbacks=[early_stopping, checkpoint]  
                            )
        print('load the best model %s to test' % bestfile)
        self.model = load_model(bestfile,custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        score = self.model.evaluate([self.validation_x_prot, self.validation_x_comp], self.validation_y)
        print("validation results is ",score)

    def fine_tune(self, pretrain_model, train_file, validation_file,  model_name):
        save_dir = os.path.join(os.getcwd(), 'models', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = load_model(pretrain_model)
        self.validation_x_prot, self.validation_x_comp, self.validation_y=dg.read_reg(validation_file)

        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(loss=self.mean_squared_error_l2, optimizer=optimizer,metrics=['mse', 'mae'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        bestfile = save_dir + "/%s_best_model.hdf5" % model_name
        checkpoint = ModelCheckpoint(bestfile, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')    
        # AUC
        RocAuc = AUC.RocAucEvaluation(validation_data=([self.validation_x_prot, self.validation_x_comp],
                                                       self.validation_y), interval=1)

        history = self.model.fit_generator(dg.read_reg_generator(train_file, self.batch_size, ft_flag = True),
                            steps_per_epoch=57,
                            epochs=self.epochs,
                            validation_data=([self.validation_x_prot, self.validation_x_comp], self.validation_y),
                            callbacks=[early_stopping, checkpoint] 
                            )
        print('load the best model %s to test' % bestfile)
        self.model = load_model(bestfile,custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        score = self.model.evaluate([self.validation_x_prot, self.validation_x_comp], self.validation_y)
        print("validation results is ",score)


    def load_model_predict(self,model_file,file):
        #read test data
        x_prot, x_comp, y = dg.read_reg(file)
        #load model
        self.model = load_model(model_file,custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        print(model_file, "load succeed!")
        predict_result = self.model.predict([x_prot, x_comp])
        tmp= model_file.split('/')[-1]
        if re.findall(r"(.+?).hdf5", tmp)==[]:
            model_name = re.findall(r"(.+?).h5", tmp)[0]
        else:
            model_name=re.findall(r"(.+?).hdf5", tmp)[0]
        # saving predict value
        df1 = pd.DataFrame(predict_result, columns=['predicted'])
        if not os.path.exists('predict_value'):
            os.mkdir('predict_value')
        df1.to_csv('predict_value/%s_predict_result.csv' % model_name, index=False)

def main():
    # train
    if len(sys.argv) == 3:
        train_file = sys.argv[1]
        validation_file = sys.argv[2]
        model_name = sys.argv[3]
        print("train data is", train_file)
        regression_solve = Regression_solve()
        regression_solve.train_model(train_file, validation_file,model_name)
    # finetune
    elif len(sys.argv == 4):
        pretrain_model = sys.argv[1]
        train_file = sys.argv[2]
        validation_file = sys.argv[3]
        model_name = sys.argv[4]
        print("train data is", train_file)
        regression_solve = Regression_solve()
        regression_solve.fine_tune(pretrain_model, train_file, validation_file,model_name)

    # predict
    else:
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        print("test file is", test_file)
        regression_solve = Regression_solve()
        regression_solve.load_model_predict(model_file, test_file)

if __name__ == '__main__':
    main()