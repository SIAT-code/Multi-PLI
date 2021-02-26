import keras

from keras.models import Model
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from keras import losses
import os
import tensorflow as tf
from keras import backend as K
import DataGenerator as dg
import get_modelv2_3
import get_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from keras.models import load_model
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
import scipy
import warnings
import sys
from sklearn.metrics import roc_curve, auc
import time
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'


class multi_task_training_respect:
    def __init__(self):
        self.model = keras.Model()
        self.model_class_task= keras.Model()
        self.model_reg_task= keras.Model()
        self.lr1 = 0.0001  
        self.lr2 = 0.0001 
        self.alpha = 0.5
        self.patience_class = 6
        self.patience_reg = 6
        self.font1 = {
            'weight': 'normal',
            'size': 16,
        }
        self.font2 = {
            'weight': 'normal',
            'size': 23,
        }

    def get_batch_data(self,prot, comp, y, batch_count, batch_size, batch_count_per_epoch):
        batch_count = batch_count % batch_count_per_epoch
        batch_prot = prot[batch_size * batch_count:min(batch_size * (batch_count + 1), len(prot))]
        batch_comp = comp[batch_size * batch_count:min(batch_size * (batch_count + 1), len(prot))]
        batch_y = y[batch_size * batch_count:min(batch_size * (batch_count + 1), len(prot))]
        return batch_prot, batch_comp, batch_y

    def draw_loss_and_accuracy_curve(self,history_class, history_class_vali, model_name, save_dir):
        train_loss = []
        vali_loss = []
        train_accuracy = []
        vali_accuracy = []
        for tmp in history_class:
            train_loss.append(tmp[0])
            train_accuracy.append(tmp[1])
        for tmp in history_class_vali:
            vali_loss.append(tmp[0])
            vali_accuracy.append(tmp[1])

        epochs = range(1, len(history_class) + 1)
        ##---------------draw loss curve------------------##
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, train_loss, 'b', label='Classification training loss')
        plt.plot(epochs, vali_loss, 'r', label='Classification validation loss')
        plt.title('Classification Training and Validation Loss', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig(save_dir + '/%s_class_training_validation_loss.png' % model_name)
        ##---------------draw accuracy curve------------------##
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, train_accuracy, 'b', label='Classification training accuracy')
        plt.plot(epochs, vali_accuracy, 'r', label='Classification validation accuracy')
        plt.title('Training and Validation Accuracy', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Accuracy', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig(save_dir + '/%s_class_training_validation_accuracy.png' % model_name)

    def draw_loss_and_mse_curve(self,history_reg, history_reg_vali, model_name, save_dir):
        train_loss = []
        vali_loss = []
        train_mse = []
        vali_mse = []
        for tmp in history_reg:
            train_loss.append(tmp[0])
            train_mse.append(tmp[1])
        for tmp in history_reg_vali:
            vali_loss.append(tmp[0])
            vali_mse.append(tmp[1])

        epochs = range(1, len(history_reg) + 1)
        ##---------------draw loss curve------------------##
        plt.figure(figsize=(10.3, 10))
        plt.plot(epochs, train_loss, 'b', label='Regression training loss')
        plt.plot(epochs, vali_loss, 'r', label='Regression validation loss')
        plt.title('Regression Training and Validation Loss', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('Loss', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig(save_dir + '/%s_reg_training_validation_loss.png' % model_name)
        ##---------------draw accuracy curve------------------##
        plt.figure(figsize=(10, 10))
        plt.plot(epochs, train_mse, 'b', label='Regression training mse')
        plt.plot(epochs, vali_mse, 'r', label='Regression validation mse')
        plt.title('Regression Training and Validation MSE', self.font2)
        plt.xlabel('Epochs', self.font2)
        plt.ylabel('MSE', self.font2)
        plt.legend(prop=self.font1)
        plt.savefig(save_dir + '/%s_reg_training_validation_mse.png' % model_name)

    def mean_squared_error_l2(self,y_true, y_pred, lmbda=0.01):
        cost = K.mean(K.square(y_pred - y_true))
        # weights = self.model.get_weights()
        weights = []
        for layer in self.model_reg_task.layers:
            # print(layer)
            weights = weights + layer.get_weights()
            # print (weights)
        result = tf.reduce_sum([tf.reduce_sum(tf.pow(wi, 2)) for wi in weights])
        l2 = lmbda * result  # K.sum([K.square(wi) for wi in weights])
        return cost + l2

    def train_model(self,class_training_file,class_validation_file,reg_training_file,reg_validation_file,model_name,
                    reg_batch_size=128,class_batch_size=128,class_epoch = 50,reg_epoch = 100,
                    pro_branch_switch1 = 'inception_block', pro_branch_switch2 = 'inception_block',
                    pro_branch_switch3='inception_block_b', pro_add_attention = False,
                    comp_branch_switch1 = 'inception_block', comp_branch_switch2 = 'inception_block',
                    comp_branch_switch3 = 'inception_block_b', comp_add_attention = False):#reg_size=256
        ##2.get_model
        save_dir = os.path.join(os.getcwd(), 'models',model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.model_class_task, self.model_reg_task = get_model.get_multi_model(save_dir, self.alpha,
            pro_branch_switch1=pro_branch_switch1,pro_branch_switch2=pro_branch_switch2,
            pro_branch_switch3=pro_branch_switch3,pro_add_attention=pro_add_attention,
            comp_branch_switch1=comp_branch_switch1,comp_branch_switch2=comp_branch_switch2,
            comp_branch_switch3=comp_branch_switch3,comp_add_attention=comp_add_attention)
        optimizer1 = keras.optimizers.Adam(lr=self.lr1)
        self.model_reg_task.compile(optimizer=optimizer1,
              loss=self.mean_squared_error_l2,#'mean_squared_error'
              metrics=['mse','mae'])
        optimizer2 = keras.optimizers.Adam(lr=self.lr2)
        self.model_class_task.compile(optimizer=optimizer2,loss='binary_crossentropy',metrics=['accuracy'])

        ##1.read data
        print("Starting read reg training data:")
        reg_train_generator = dg.read_reg_generator(reg_training_file, reg_batch_size)
        reg_vali_prot, reg_vali_comp, reg_vali_value = dg.read_reg(reg_validation_file)
        print('regression validation data shape:', len(reg_vali_prot))
        class_train_generator = dg.read_class_generator(class_training_file, class_batch_size)
        class_vali_prot, class_vali_comp, class_vali_label = dg.read_class(class_validation_file)
        print('classification validation data shape:', len(class_vali_prot))

        ##3.training model
        #before train prepare
        batch_count_of_class=0
        batch_count_per_epoch_class=189109//class_batch_size
        batch_count_of_reg = 0
        batch_count_per_epoch_reg = 18071 // reg_batch_size  
        epoch_class = 0
        epoch_reg=0
        history_class=[]
        history_class_vali=[]
        history_reg=[]
        history_reg_vali=[]
        class_erally_stop_flag=1
        reg_erally_stop_flag = 1
        class_batch_count = class_epoch * batch_count_per_epoch_class
        reg_batch_count = reg_epoch * batch_count_per_epoch_reg
        K = reg_batch_count/class_batch_count
        total_batch_count=class_batch_count+reg_batch_count

        #start train
        reg_min_loss = float('inf')
        reg_min_loss_index = 0
        class_min_loss=float('inf')
        class_min_loss_index=0

        best_reg_model = None
        best_class_model = None
        best_reg_file =  save_dir + "/%s_best_reg_model.hdf5" % model_name
        best_class_file = save_dir + "/%s_best_class_model.hdf5" % model_name
        reg_loss=[]
        class_loss=[]
        for i in range(total_batch_count):
                #regression
                if np.random.rand() * (1+K) >= 1 and reg_erally_stop_flag and epoch_reg<reg_epoch:
                    print('batch %d(reg):'%i)
                    reg_batch_prot, reg_batch_comp, reg_batch_value = next(reg_train_generator)
                    tmp_loss=self.model_reg_task.train_on_batch([reg_batch_prot, reg_batch_comp], reg_batch_value)
                    reg_loss.append(tmp_loss)
                    batch_count_of_reg+=1
                    if batch_count_of_reg % batch_count_per_epoch_reg==0 and batch_count_of_reg>0:
                        epoch_reg += 1
                        print("regression epoch %d:"%epoch_reg)
                        #train performance:loss, mse, mae
                        print('    regression training loss=',np.mean(reg_loss,axis=0))
                        history_reg.append(np.mean(reg_loss,axis=0))
                        reg_loss=[]
                        #validation performance
                        score=self.model_reg_task.evaluate([reg_vali_prot,reg_vali_comp],reg_vali_value)
                        print('    regression evaluation loss=',score)
                        history_reg_vali.append(score)
                        #checkpoint and earlly stop
                        if epoch_reg-reg_min_loss_index>=self.patience_reg:
                            reg_erally_stop_flag=0
                        if score[0]<reg_min_loss:
                            reg_min_loss_index=epoch_reg
                            reg_min_loss=score[0]
                            #checkpoint
                            best_reg_model = self.model_reg_task
                # classification
                else:
                    if class_erally_stop_flag and epoch_class<class_epoch:
                        print('batch %d(class):' % i)
                        class_batch_prot, class_batch_comp, class_batch_label = next(class_train_generator)
                        tmp_loss=self.model_class_task.train_on_batch([class_batch_prot, class_batch_comp], class_batch_label)
                        class_loss.append(tmp_loss)
                        batch_count_of_class += 1
                        if batch_count_of_class % batch_count_per_epoch_class == 0 and batch_count_of_class>0:
                            epoch_class += 1
                            print("classification epoch %d:"%epoch_class)
                            # train performance:loss, mse, mae
                            print('    classification training loss=',np.mean(class_loss,axis=0))
                            history_class.append(np.mean(class_loss,axis=0))
                            class_loss=[]#
                            accuracy = self.model_class_task.evaluate([class_vali_prot, class_vali_comp], class_vali_label)
                            # validation performance
                            print('    classification evaluation loss=',accuracy)
                            history_class_vali.append(accuracy)
                            # checkpoint and earlly stop
                            if epoch_class - class_min_loss_index >= self.patience_class:
                                class_erally_stop_flag = 0
                            if accuracy[0] < class_min_loss:
                                class_min_loss_index = epoch_class
                                class_min_loss = accuracy[0]
                                # checkpoint
                                best_class_model = self.model_class_task

        ##5.save model
        #(1).class model
        model_path = os.path.join(save_dir,model_name+'_class.h5')
        best_class_model.save(model_path)
        #(2).reg model
        model_path = os.path.join(save_dir,model_name+'_reg.h5')
        best_reg_model.save(model_path)
        print("save model!")

    def save_predict_result(self,predict_result,real_label_or_value,model_name,class_or_reg,type):

        if predict_result.shape[1] == 1:
            if class_or_reg=='class':
                df = predict_result
                df.columns = ['predict_label']
            else:
                df = predict_result
                df.columns = ['predict_value']
        else:
            df = predict_result
            df.columns = ['predict_label','predict_value']
        if class_or_reg=='class':
            df['real_lable'] = real_label_or_value
        else:
            df['real_value'] = real_label_or_value
        df['set']=type
        if not os.path.exists('predict_value'):
            os.mkdir('predict_value')
        df.to_csv('predict_value/multi-task_model_%s_%s_%s_predict_result.csv' % (model_name,class_or_reg,type),
                  index=False)
        print('predict_value/multi-task_model_%s_%s_%s_predict_result.csv has been saved!' % (model_name,class_or_reg,type))
        return df

    def computer_parameter_draw_scatter_plot(self,predictions, model_name):
        sns.set(context='paper', style='white')
        sns.set_color_codes()
        set_colors = {'train': 'b', 'validation': 'green', 'test': 'purple'}
        for set_name, table in predictions.groupby('set'):
            rmse = ((table['predict_value'] - table['real_value']) ** 2).mean() ** 0.5
            mae = (np.abs(table['predict_value'] - table['real_value'])).mean()
            corr = scipy.stats.pearsonr(table['predict_value'], table['real_value'])
            lr = LinearRegression()
            lr.fit(table[['predict_value']], table['real_value'])
            y_ = lr.predict(table[['predict_value']])
            sd = (((table["real_value"] - y_) ** 2).sum() / (len(table) - 1)) ** 0.5
            print("%10s set: RMSE=%.3f, MAE=%.3f, R=%.2f (p=%.2e), SD=%.3f" %
                  (set_name, rmse, mae, *corr, sd))

            grid = sns.jointplot('real_value', 'predict_value', data=table, stat_func=None, color=set_colors[set_name],
                                 space=0, size=4, ratio=4, s=20, edgecolor='w', ylim=(0, 16), xlim=(0, 16))  # (0.16)
            grid.set_axis_labels('real', 'predicted')#, fontsize=16
            grid.ax_joint.set_xticks(range(0, 16, 5))
            grid.ax_joint.set_yticks(range(0, 16, 5))

            a = {'train': 'training', 'validation': 'validation', 'test': 'test'}
            set_name=a[set_name]
            grid.ax_joint.text(1, 14, set_name + ' set', fontsize=14)  # 调整标题大小
            grid.ax_joint.text(16, 19.5, 'RMSE: %.3f' % (rmse), fontsize=9)
            grid.ax_joint.text(16, 18.5, 'MAE: %.3f ' % mae, fontsize=9)
            grid.ax_joint.text(16, 17.5, 'R: %.2f ' % corr[0], fontsize=9)
            grid.ax_joint.text(16, 16.5, 'SD: %.3f ' % sd, fontsize=9)

            grid.fig.savefig('%s_%s_scatter_plot.jpg' %(model_name,set_name), dpi=400)

    def draw_ROC_curve(self,predictions, model_name):
        set_colors = {'train': 'b', 'validation': 'green', 'test': 'purple','independent test':'r'}
        for set_name, table in predictions.groupby('set'):
            fpr, tpr, threshold = roc_curve(table['real_lable'],table['predict_label'])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 10))
            lw = 2
            plt.plot(fpr, tpr, color=set_colors[set_name],
                     lw=lw, label='ROC curve (auc = %0.3f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'b--', lw=lw,
                     label='Random guess (auc = 0.5)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.tick_params(labelsize=20)
            plt.xlabel('False Positive Rate', self.font2)
            plt.ylabel('True Positive Rate', self.font2)
            # plt.title('ROC curv')
            plt.legend(loc="lower right", prop=self.font1)
            plt.savefig("%s_%s_ROC_curve.png" %(model_name,set_name))

    def test_model(self,model_file,class_test_file,class_train_file,class_vali_file,
                   reg_test_file,reg_train_file,reg_vali_file):
        ##read data
        print('starting read data!')
        #1.train data
        class_train_prot, class_train_comp, class_train_label=dg.multi_process_read_pro_com_file(class_train_file)
        reg_train_prot, reg_train_comp,_, reg_train_value=dg.multi_process_read_pro_com_file_regression(reg_train_file)
        #2.validation data
        class_vali_prot, class_vali_comp, class_vali_label = dg.multi_process_read_pro_com_file(class_vali_file)
        reg_vali_prot, reg_vali_comp, _,reg_vali_value = dg.multi_process_read_pro_com_file_regression(reg_vali_file)
        #3.test data
        class_test_prot, class_test_comp, class_test_label = dg.multi_process_read_pro_com_file(class_test_file)
        reg_test_prot, reg_test_comp,_, reg_test_value = dg.multi_process_read_pro_com_file_regression(reg_test_file)
        print('classification data size:', len(class_train_prot), len(class_vali_prot), len(class_test_prot))
        print('regression data size:', len(reg_train_prot),len(reg_vali_prot),len(reg_test_prot))

        ##load_model
        print('loading modle!')
        model = load_model(model_file)
        tmp = model_file.split('/')[-1]
        model_name = re.findall(r"(.+?).h5", tmp)[0]

        ## saving predict value
        #predict value
        #1.train
        class_train_predict_value = model.predict([class_train_prot, class_train_comp])
        class_train_predict_value_df=pd.DataFrame(class_train_predict_value[0],columns=['label'])
        class_train_predict_value_df['value']=class_train_predict_value[1]
        reg_train_predict_value = model.predict([reg_train_prot, reg_train_comp])
        reg_train_predict_value_df=pd.DataFrame(reg_train_predict_value[0],columns=['label'])
        reg_train_predict_value_df['value']=reg_train_predict_value[1]
        #2.vali
        class_vali_predict_value = model.predict([class_vali_prot, class_vali_comp])
        class_vali_predict_value_df = pd.DataFrame(class_vali_predict_value[0])
        class_vali_predict_value_df['value']=class_vali_predict_value[1]
        reg_vali_predict_value = model.predict([reg_vali_prot, reg_vali_comp])
        reg_vali_predict_value_df = pd.DataFrame(reg_vali_predict_value[0])
        reg_vali_predict_value_df['value']=reg_vali_predict_value[1]
        #3.test
        class_test_predict_value = model.predict([class_test_prot, class_test_comp])
        class_test_predict_value_df = pd.DataFrame(class_test_predict_value[0])
        class_test_predict_value_df['value']=class_test_predict_value[1]
        reg_test_predict_value=model.predict([reg_test_prot, reg_test_comp])
        reg_test_predict_value_df = pd.DataFrame(reg_test_predict_value[0])
        reg_test_predict_value_df['value']=reg_test_predict_value[1]
        # save predicted value
        #1
        class_train_df = self.save_predict_result(class_train_predict_value_df, class_train_label, model_name, 'class', 'train')
        reg_train_df = self.save_predict_result(reg_train_predict_value_df, reg_train_value, model_name, 'reg', 'train')
        #2
        class_vali_df = self.save_predict_result(class_vali_predict_value_df, class_vali_label, model_name, 'class', 'validation')
        reg_vali_df = self.save_predict_result(reg_vali_predict_value_df, reg_vali_value, model_name, 'reg', 'validation')
        #3
        class_test_df = self.save_predict_result(class_test_predict_value_df, class_test_label, model_name, 'class', 'test')
        reg_test_df = self.save_predict_result(reg_test_predict_value_df, reg_test_value, model_name, 'reg', 'test')

        ## computing parameters and drawing scatter plot
        self.computer_parameter_draw_scatter_plot(reg_train_df, model_name)
        self.computer_parameter_draw_scatter_plot(reg_vali_df, model_name)
        self.computer_parameter_draw_scatter_plot(reg_test_df, model_name)
        self.draw_ROC_curve(class_train_df, model_name)
        self.draw_ROC_curve(class_vali_df, model_name)
        self.draw_ROC_curve(class_test_df, model_name)

    def reg_test_model(self,model_file,reg_test_file,reg_train_file=None,reg_vali_file=None):
        ##load_model
        print('loading modle!')
        self.model_reg_task = load_model(model_file,
                                         custom_objects={'mean_squared_error_l2': self.mean_squared_error_l2})
        tmp = model_file.split('/')[-1]
        if tmp.find('.h5')!=-1:
            model_name = re.findall(r"(.+?).h5", tmp)[0]
        else:
            model_name = re.findall(r"(.+?).hdf5", tmp)[0]
        ##1.read data
        print('starting read data!')
        reg_test_prot, reg_test_comp,_, reg_test_value = dg.read_pro_com_file_regression(reg_test_file)#multi_process_read_pro_com_file_regression(reg_test_file)
        print('test data size:',len(reg_test_prot))

        reg_test_predict_value=self.model_reg_task.predict([reg_test_prot, reg_test_comp])
        if model_name[-3:]=='reg':#reg_model
            reg_test_predict_value_df = pd.DataFrame(reg_test_predict_value,columns=['value'])
        else:#total model
            reg_test_predict_value_df = pd.DataFrame(reg_test_predict_value[0], columns=['label'])
            reg_test_predict_value_df['value']=reg_test_predict_value[1]
        reg_test_df = self.save_predict_result(reg_test_predict_value_df, reg_test_value, model_name, 'reg', 'test')
        self.computer_parameter_draw_scatter_plot(reg_test_df, model_name)
        if reg_train_file!=None:
            reg_train_prot, reg_train_comp,_, reg_train_value = dg.multi_process_read_pro_com_file_regression(reg_train_file)
            reg_train_predict_value = self.model_reg_task.predict([reg_train_prot, reg_train_comp])
            reg_train_predict_value=pd.DataFrame(reg_train_predict_value)
            reg_train_df = self.save_predict_result(reg_train_predict_value, reg_train_value, model_name, 'reg', 'train')
            self.computer_parameter_draw_scatter_plot(reg_train_df, model_name)
        if reg_vali_file!=None:
            reg_vali_prot, reg_vali_comp,_, reg_vali_value = dg.multi_process_read_pro_com_file_regression(reg_vali_file)
            #predict value
            reg_vali_predict_value = self.model_reg_task.predict([reg_vali_prot, reg_vali_comp])
            reg_vali_predict_value=pd.DataFrame(reg_vali_predict_value)
            reg_vali_df = self.save_predict_result(reg_vali_predict_value, reg_vali_value, model_name, 'reg', 'validation')
            self.computer_parameter_draw_scatter_plot(reg_vali_df, model_name)

    def class_test_model(self,model_file,class_test_file,class_train_file=None,class_vali_file=None):
        ##load_model
        print('loading modle!')
        self.model_class_task = load_model(model_file)
        tmp = model_file.split('/')[-1]
        if tmp.find('.h5')!=-1:
            model_name = re.findall(r"(.+?).h5", tmp)[0]
        else:
            model_name = re.findall(r"(.+?).hdf5", tmp)[0]
        # 1. data
        ##read data
        print('starting read data!')
        if class_test_file.split('/')[2]=='one-hot_dataset4':
            print("对dataset4进行特殊对待！")
            class_test_prot, class_test_comp, class_test_label,class_test_value = dg.multi_process_read_pro_com_file_regression(class_test_file)
            # df_tmp = pd.DataFrame(class_test_value, columns=['real_value'])
            # df_tmp.to_csv('dataset4_real_value.csv')
        else:
            class_test_prot, class_test_comp, class_test_label = dg.multi_process_read_pro_com_file(class_test_file)
        print('test data size:', len(class_test_prot))
        # 2.predict value
        class_test_predict_value = self.model_class_task.predict([class_test_prot, class_test_comp])
        print(model_name[-9:-4] )
        if model_name[-9:-4] == 'class':  # class_model
            class_test_predict_value_df = pd.DataFrame(class_test_predict_value, columns=['label'])
        else:  # total model
            class_test_predict_value_df = pd.DataFrame(class_test_predict_value[0], columns=['label'])
            class_test_predict_value_df['value'] = class_test_predict_value[1]
        # 3.save predicted value
        test_file_name = re.findall(r"(.+?).txt", class_test_file)[0].split('/')[-1]
        class_test_df = self.save_predict_result(class_test_predict_value_df, class_test_label, model_name+test_file_name, 'class', 'test')
        # 4.computing parameters and drawing auc plot
        self.draw_ROC_curve(class_test_df, model_name+test_file_name)
        if class_train_file!=None:
            # 1.read data
            class_train_prot, class_train_comp, class_train_label = dg.multi_process_read_pro_com_file(class_train_file)
            print('train data size:',len(class_train_prot))
            # 2.predict value
            class_train_predict_value = self.model_class_task.predict([class_train_prot, class_train_comp])
            if model_name[-4:] == 'class':  # reg_model
                class_train_predict_value_df = pd.DataFrame(class_train_predict_value, columns=['label'])
            else:  # total model
                class_train_predict_value_df = pd.DataFrame(class_train_predict_value[0], columns=['label'])
                class_train_predict_value_df['value'] = class_train_predict_value[1]
            # 3.save predicted value
            class_train_df = self.save_predict_result(class_train_predict_value_df, class_train_label, model_name, 'class', 'train')
            # 4.computing parameters and drawing auc plot
            self.draw_ROC_curve(class_train_df, model_name)
        if class_vali_file != None:
            # 1.read data
            class_vali_prot, class_vali_comp, class_vali_label = dg.multi_process_read_pro_com_file(class_vali_file)
            print('validation data size:', len(class_vali_prot))
            # 2.predict value
            class_vali_predict_value = self.model_class_task.predict([class_vali_prot, class_vali_comp])
            if model_name[-4:] == 'class':  # reg_model
                class_vali_predict_value_df = pd.DataFrame(class_vali_predict_value, columns=['label'])
            else:  # total model
                class_vali_predict_value_df = pd.DataFrame(class_vali_predict_value[0], columns=['label'])
                class_vali_predict_value_df['value'] = class_vali_predict_value[1]
            # 3.save predicted value
            class_vali_df = self.save_predict_result(class_vali_predict_value_df, class_vali_label, model_name, 'class',
                                                'validation')
            # 4.computing parameters and drawing auc plot
            self.draw_ROC_curve(class_vali_df, model_name)

    def load_model_predict(self,model_file,file):
        ##read data
        print('starting read data!')
        x_prot, x_comp, y_label = dg.read_pro_com_file(file)
        # class_test_prot, class_test_comp, class_test_label = dg.multi_process_read_pro_com_file(class_test_file)
        print('data size:', len(x_prot), len(x_comp), len(y_label))

        ##load_model
        print('loading modle!')
        model = load_model(model_file)
        tmp = model_file.split('/')[-1]
        model_name = re.findall(r"(.+?).h5", tmp)[0]

        ## saving predict value
        #predict value
        predict_value = model.predict([x_prot, x_comp])
        predict_value_df = pd.DataFrame(predict_value[0])
        predict_value_df['value']=predict_value[1]
        # save predicted value
        self.save_predict_result(predict_value_df, y_label, model_name, 'class', 'test')


if __name__=='__main__':

    if len(sys.argv)==6:
        # train_model()
        class_training_file=sys.argv[1]
        class_validation_file=sys.argv[2]
        reg_training_file=sys.argv[3]
        reg_validation_file=sys.argv[4]
        model_name=sys.argv[5]
        print('Start training model:')
        multi_task_solve=multi_task_training_respect()
        multi_task_solve.train_model(class_training_file,class_validation_file,reg_training_file,reg_validation_file,model_name)
    elif len(sys.argv)==8:
        #test model
        model_file=sys.argv[1]
        class_test_file  = sys.argv[2]
        class_train_file = sys.argv[3]
        class_vali_file = sys.argv[4]
        reg_test_file= sys.argv[5]
        reg_train_file =sys.argv[6]
        reg_vali_file=sys.argv[7]
        print('Start test model:')
        multi_task_solve = multi_task_training_respect()
        multi_task_solve.test_model(model_file,
                                    class_test_file,class_train_file,class_vali_file,
                                    reg_test_file,reg_train_file,reg_vali_file)
    elif len(sys.argv)==5:
        #test model
        model_file=sys.argv[1]
        reg_test_file = sys.argv[2]
        reg_train_file= sys.argv[3]
        reg_vali_file=sys.argv[4]
        print('test model regression performance:')
        multi_task_solve = multi_task_training_respect()
        multi_task_solve.reg_test_model(model_file, reg_test_file, reg_train_file, reg_vali_file)
    elif len(sys.argv)==3:
        # test model
        model_file = sys.argv[1]
        class_test_file = sys.argv[2]
        print('test model classification performance:')
        multi_task_solve = multi_task_training_respect()
        multi_task_solve.class_test_model(model_file, class_test_file)
    else:
        print("input parametes not illegal, please check and reinput!")