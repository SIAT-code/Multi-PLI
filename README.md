# Multi-PLI

## Requirement
```
python 3.7
tensorflow 2.1.0
keras 2.3.1
numpy
pandas 
scikit-learn  
```

## Usage 
To train an classification model, use the classification_solve.py script. For example:
```
  usage: classification_solve.py train_file vaild_file model_name
```
To finetune an classification model, use the classification_solve.py script. For example:
```
  usage: classification_solve.py pretrain_model train_file vaild_file model_name
```
To train a regression model, use the regression_solve.py script. For example:
```
  usage: regression_solve.py train_file vaild_file model_name
```
To train a multi-task model, use the multi-task_training_reaspective.py script. For example:
```
  usage: multi-task_training_reaspective.py class_train_file class_valid_file reg_train_file reg_valid_file model_name
```
