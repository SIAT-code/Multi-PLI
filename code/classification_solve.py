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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  
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

class Classification_solve():
    def __init__(self,n=8,batch_size=128,epochs=300,lr=0.0001,patience=10,alpha=0.5):
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
        print('Established a Classification_solve object.')

    def draw_loss_change(self,history,model_name, save_dir):
        history_dict = history.history
        loss_values = history_dict['loss']
        if 'val_loss' in history_dict:
            val_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, loss_values, 'b', label='Training loss')
        if 'val_loss' in history_dict:
            plt.plot(epochs, val_loss_values, 'r', label='Validation loss')
        plt.title('Training and validation loss', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        save_path = save_dir + '/%s_training_validation_loss.png'%model_name
        plt.savefig(save_path)

    def draw_ROC_curve(self,type,model_name, save_dir):
        if type=='validation':
            y_predict = self.model.predict([self.validation_x_prot, self.validation_x_comp])
            y_true=self.validation_y
        elif type=='train':
            y_predict = self.model.predict([self.train_x_prot, self.train_x_comp])
            y_true = self.train_y
        else:
            y_predict = self.model.predict([self.test_x_prot, self.test_x_comp])
            y_true=self.test_y
        # Compute ROC curve and ROC area for each class
        fpr, tpr, threshold = roc_curve(y_true,y_predict )  
        roc_auc = auc(fpr, tpr) 
        print("auc:", roc_auc)
        plt.figure(figsize=(10, 10))
        lw = 2
        set_colors = {'train': 'b','validation': 'green', 'test': 'purple'}
        plt.plot(fpr, tpr, color=set_colors[type],
                 lw=lw, label='ROC curve (auc = %0.3f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='brown', lw=lw,
                 label='Random guess (auc = 0.5)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.tick_params(labelsize=20)
        plt.xlabel('False Positive Rate', self.font2)
        plt.ylabel('True Positive Rate', self.font2)
        # plt.title('ROC curv')
        plt.legend(loc="lower right", prop=self.font1)
        save_path = save_dir + "/%s_%s_ROC_curve.png" % (model_name,type)
        plt.savefig(save_path)

    def save_predict_result(self,type,model_name, save_dir):
        if type=='test':
            predict_result = self.model.predict([self.test_x_prot, self.test_x_comp])
            real=self.test_y
        elif type=='validation':
            predict_result = self.model.predict([self.validation_x_prot, self.validation_x_comp])
            real = self.validation_y
        else:
            predict_result = self.model.predict([self.train_x_prot, self.train_x_comp])
            real = self.train_y
        df1 = pd.DataFrame(predict_result, columns=['predicted'])
        df1['real'] = real
        df1['set'] = type
        save_path = save_dir + '/%s_%s_predict_result.csv' % (model_name,type)
        df1.to_csv(save_path, index=False)

    def train_model(self, train_file, validation_file,  model_name):
        save_dir = os.path.join(os.getcwd(), 'models', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = get_model.get_model_classification(save_dir,self.alpha,                 
                    pro_branch_switch1 = 'inception_block', pro_branch_switch2 = 'inception_block',
                    pro_branch_switch3='inception_block_b', pro_add_attention = False,
                    comp_branch_switch1 = 'inception_block', comp_branch_switch2 = 'inception_block',
                    comp_branch_switch3 = 'inception_block_b', comp_add_attention = False)
        self.validation_x_prot, self.validation_x_comp, self.validation_y=dg.read_class(validation_file)

        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        bestfile = save_dir + "/%s_best_model.hdf5" % model_name
        checkpoint = ModelCheckpoint(bestfile, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')    
        # AUC
        RocAuc = AUC.RocAucEvaluation(validation_data=([self.validation_x_prot, self.validation_x_comp],
                                                       self.validation_y), interval=1)

        history = self.model.fit_generator(dg.read_class_generator(train_file, self.batch_size),
                            steps_per_epoch=739,
                            epochs=self.epochs,
                            validation_data=([self.validation_x_prot, self.validation_x_comp], self.validation_y),
                            callbacks=[RocAuc,early_stopping, checkpoint]  
                            )
        print('load the best model %s to test' % bestfile)
        self.model = load_model(bestfile)
        results = self.model.evaluate([self.validation_x_prot, self.validation_x_comp], self.validation_y)
        print('validation accuracy:', results)
        self.draw_loss_change(history,model_name, save_dir) 
        self.draw_ROC_curve('validation',model_name, save_dir)
        self.save_predict_result('validation',model_name, save_dir)

    def fine_tune(self, pretrain_model, train_file, validation_file,  model_name):
        save_dir = os.path.join(os.getcwd(), 'models', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model = load_model(pretrain_model)
        self.validation_x_prot, self.validation_x_comp, self.validation_y=dg.read_class(validation_file)

        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.patience)
        bestfile = save_dir + "/%s_best_model.hdf5" % model_name
        checkpoint = ModelCheckpoint(bestfile, monitor='val_loss', verbose=1, save_best_only=True,
                                     mode='min')    
        # AUC
        RocAuc = AUC.RocAucEvaluation(validation_data=([self.validation_x_prot, self.validation_x_comp],
                                                       self.validation_y), interval=1)

        history = self.model.fit_generator(dg.read_class_generator(train_file, self.batch_size, ft_flag = True),
                            steps_per_epoch=739,
                            epochs=self.epochs,
                            validation_data=([self.validation_x_prot, self.validation_x_comp], self.validation_y),
                            callbacks=[RocAuc,early_stopping, checkpoint]  
                            )
        print('load the best model %s to test' % bestfile)
        self.model = load_model(bestfile)
        results = self.model.evaluate([self.validation_x_prot, self.validation_x_comp], self.validation_y)
        print('validation accuracy:', results)
        self.draw_loss_change(history,model_name, save_dir) 
        self.draw_ROC_curve('validation',model_name, save_dir)
        self.save_predict_result('validation',model_name, save_dir)
        

    def load_model_predict(self,model_file,filename):
        #read test data
        x_prot, x_comp = self.read_class_nolabel(filename)
        #load model
        self.model = load_model(model_file)
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
        classification_solve = Classification_solve()
        classification_solve.train_model(train_file, validation_file,model_name)
    # finetune
    elif len(sys.argv == 4):
        pretrain_model = sys.argv[1]
        train_file = sys.argv[2]
        validation_file = sys.argv[3]
        model_name = sys.argv[4]
        print("train data is", train_file)
        classification_solve = Classification_solve()
        classification_solve.fine_tune(pretrain_model, train_file, validation_file,model_name)

    # predict
    else:
        model_file = sys.argv[1]
        test_file = sys.argv[2]
        print("test file is", test_file)
        classification_solve = Classification_solve()
        classification_solve.load_model_predict(model_file, test_file)

if __name__ == '__main__':
    main()