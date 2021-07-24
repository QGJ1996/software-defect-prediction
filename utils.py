from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout,Input,BatchNormalization,Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import keras.backend as K
import math
from sklearn.metrics import roc_curve


GPUS = tf.config.experimental.list_physical_devices("GPU")
for GPU in GPUS:
    tf.config.experimental.set_memory_growth(GPU,True)
def resample(data,label,rate):
    number_one = len(np.where(label==1)[0])
    number_zero = len(np.where(label==0)[0])
    if number_one/number_zero > rate:
        return data,label
    sample_one = int(len(label) * rate / (1 + rate))
    sample_zero = len(label) - sample_one
    overSampler = RandomOverSampler(sampling_strategy={0:number_zero,1:sample_one},random_state=111)
    data_sample,label_ = overSampler.fit_resample(data,label)
    underSampler = RandomUnderSampler(sampling_strategy={0:sample_zero,1:sample_one},random_state=111,replacement=False)
    data_sample,label_ = underSampler.fit_resample(data_sample,label_)
    return (data_sample,label_)


def buil_model(input_size, lr, d_rate, loss_fn, metrics_fn):
    K.clear_session()
    inputs = Input(shape=[input_size, ])
    hidden = Dense(units=input_size * 2, activation="relu", kernel_regularizer=regularizers.l2(0.01))(inputs)
    norm = BatchNormalization()(hidden)
    drop = Dropout(d_rate)(norm)
    # hidden = Dense(units=input_size,activation="relu")(drop)
    # norm = BatchNormalization()(hidden)

    out = Dense(units=1, activation="sigmoid")(drop)
    opt = Adam(learning_rate=lr)
    model = Model(inputs, out)
    model.compile(loss=loss_fn, optimizer=opt, metrics=metrics_fn)
    return model

def train_model(model,train_data,train_label,save_filepath,epochs):
    callback_list = [
        tf.keras.callbacks.EarlyStopping(monitor="loss",patience=10),
        tf.keras.callbacks.ModelCheckpoint(filepath=save_filepath,monitor="loss",save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',factor=0.1,patience=5)
    ]
    model.fit(train_data,train_label,epochs=epochs,batch_size=64,callbacks=callback_list,verbose=False)
    return model

def getFPR(predict,y_true):
    eplision = 1e-7
    FP = np.sum(np.multiply((1 - y_true), predict))
    TN = np.sum(np.multiply((1 - y_true), (1 - np.array(predict))))
    return FP/(FP+TN+eplision)

def getBalance(y_true,y_score):
    fpr,tpr,thresholds = roc_curve(y_true=y_true,y_score=y_score)
    FPR = fpr[int(len(fpr)/2)]
    TPR = tpr[int(len(fpr)/2)]
    Balance = 1 - math.sqrt((1 - TPR) ** 2 + FPR ** 2)
    return Balance
def precision(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    epsilon = tf.convert_to_tensor(K.common._EPSILON)
    TP = tf.reduce_sum(tf.multiply(y_true,tf.round(y_pred)))
    FP = tf.reduce_sum(tf.multiply((tf.ones_like(y_true) - y_true),tf.round(y_pred)))
    return TP/(TP+FP+epsilon)

def recall(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    epsilon = tf.convert_to_tensor(K.common._EPSILON)
    TP = tf.reduce_sum(tf.multiply(y_true,tf.round(y_pred)))
    FN = tf.reduce_sum(tf.multiply(y_true,(tf.ones_like(y_pred) - tf.round(y_pred))))
    return TP/(TP+FN+epsilon)