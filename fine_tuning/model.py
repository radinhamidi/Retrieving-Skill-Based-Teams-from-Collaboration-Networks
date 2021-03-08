from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Lambda
from keras.losses import mse, binary_crossentropy, mae, kld, categorical_crossentropy
import time
import csv
from keras.callbacks import Callback
import pickle as pkl
from keras.callbacks import EarlyStopping
from keras_metrics.metrics import true_negative
from py.builtin import enumerate
from tornado.autoreload import watch
import cmn.utils
from keras.layers import Input, Dense, Embedding, Flatten
from keras.models import Model
from contextlib import redirect_stdout
import cmn.utils
from cmn.utils import *
import dal.load_dblp_data as dblp
import eval.evaluator as dblp_eval
import eval.ranking as rk
import ml_metrics as metrics
from cmn.variational import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class watcher(Callback):
    def on_train_begin(self, logs={}):
        self.intervals = []
        self.ndcg = []
        self.map = []
        self.sum = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.sum += time.time() - self.epoch_time_start
        if epoch < 30:
            recorder_step = 5
        elif epoch < 300:
            recorder_step = 50
        else:
            recorder_step = 150
        if epoch%recorder_step == 0:
            self.intervals.append(self.sum)
            self.sum = 0
            y_true = y_test
            y_pred = autoencoder.predict(x_test)
            pred_index, true_index = dblp_eval.find_indices(y_pred, y_true)
            self.ndcg.append(ndcg_metric(pred_index, true_index))
            self.map.append(map_metric(pred_index, true_index))


watchDog = watcher()


def ndcg_metric(pred_index, true_index):
    return np.mean([rk.ndcg_at(pred_index, true_index, k=5), rk.ndcg_at(pred_index, true_index, k=10)])


def map_metric(pred_index, true_index):
    return np.mean([metrics.mapk(true_index, pred_index, k=5), metrics.mapk(true_index, pred_index, k=10)])


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0.0001)

#running settings
dataset_name = 'DBLP'
method_name = 'M2V_TeamFormation'

#eval settings
k_fold = 10
k_max = 100 #cut_off for eval
evaluation_k_set = np.arange(1, k_max+1, 1)

#nn settings
epochs = 2000
back_propagation_batch_size = 32
training_batch_size = 6000
min_skill_size = 0
min_member_size = 0
latent_dim = 2
beta = 30

print(tf.test.is_gpu_available())
m2v_path = '../dataset/embedding_dict.pkl'


if dblp.ae_data_exist(file_path='../dataset/ae_e_m2v_tSkill_dataset.pkl'):
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_e_m2v_tSkill_dataset.pkl')
else:
    if not dblp.ae_data_exist(file_path='../dataset/ae_dataset.pkl'):
        dblp.extract_data(filter_journals=True, skill_size_filter=min_skill_size, member_size_filter=min_member_size, output_dir='../dataset/ae_dataset.pkl')
    if not dblp.preprocessed_dataset_exist(file_path='../dataset/dblp_preprocessed_dataset.pkl') or not dblp.train_test_indices_exist(file_path='../dataset/Train_Test_indices.pkl'):
        dblp.dataset_preprocessing(dblp.load_ae_dataset(file_path='../dataset/ae_dataset.pkl'), indices_dict_file_path='../dataset/Train_Test_indices.pkl', preprocessed_dataset_file_path='../dataset/dblp_preprocessed_dataset.pkl', seed=seed, kfolds=k_fold)
    preprocessed_dataset = dblp.load_preprocessed_dataset(file_path='../dataset/dblp_preprocessed_dataset.pkl')

    dblp.nn_m2v_embedding_dataset_generator(model_path=m2v_path, dataset=preprocessed_dataset, output_file_path='../dataset/ae_e_m2v_tSkill_dataset.pkl', mode='skill', max_length=22)
    del preprocessed_dataset
    dataset = dblp.load_ae_dataset(file_path='../dataset/ae_e_m2v_tSkill_dataset.pkl')



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


train_test_indices = dblp.load_train_test_indices(file_path='../dataset/Train_Test_indices.pkl')


# k_fold Cross Validation
cvscores = []

# Defining evaluation scores holders for train data
r_at_k_all_train = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall_train = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set
mapk_train = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set

# Defining evaluation scores holders for test data
r_at_k_all = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
r_at_k_overall = dblp_eval.init_eval_holder(evaluation_k_set)  # overall r@k of instances in one fold and one k_evaluation_set
mapk = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
ndcg = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
mrr = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set
tf_score = dblp_eval.init_eval_holder(evaluation_k_set)  # all r@k of instances in one fold and one k_evaluation_set

load_weights_from_file_q = input('Load weights from file? (y/n)')
more_train_q = input('Train more? (y/n)')

time_str = time.strftime("%Y_%m_%d-%H_%M_%S")
result_output_name = "../output/predictions/{}_output.csv".format(method_name)
with open(result_output_name, 'w') as file:
    writer = csv.writer(file)
    writer.writerow(
        ['Method Name', '# Total Folds', '# Fold Number', '# Predictions', '# Truth', 'Computation Time (ms)',
         'Prediction Indices', 'True Indices'])

embedding_file = pickle.load(open(m2v_path, 'rb'))
skill_embedding = embedding_file['skill']
EMBEDDING_DIM = len(skill_embedding[0])
skill_size = len(skill_embedding.keys())
print("Embedding dim is:", EMBEDDING_DIM)
embedding_matrix = np.zeros((skill_size + 1, EMBEDDING_DIM))
for i, embedding in skill_embedding.items():
    if embedding is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding

for fold_counter in range(1,k_fold+1):
    x_train, y_train, x_test, y_test = dblp.get_fold_data(fold_counter, dataset, train_test_indices)

    input_dim = x_train.shape[1]
    output_dim = y_train.shape[1]
    print("Input/output Dimensions:  ", input_dim, output_dim)

    # this is our input placeholder
    # network parameters
    intermediate_dim_encoder = input_dim
    intermediate_dim_decoder = output_dim

    # build encoder model
    inputs = Input(shape=(input_dim,), name='encoder_input')
    embedding_layer = Embedding(skill_size + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=input_dim,
                            trainable=True)(inputs)
    # x = Flatten()(embedding_layer)
    x = Lambda(lambda x: keras.backend.mean(x, axis=1))(embedding_layer)
    # x = Dense(intermediate_dim_encoder, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim_decoder, activation='relu')(latent_inputs)
    outputs = Dense(output_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    # instantiate model
    outputs = decoder(encoder(inputs)[2])
    autoencoder = Model(inputs, outputs, name='vae_mlp')

    models = (encoder, decoder)

    def vae_loss(y_true, y_pred):
        reconstruction_loss = mse(y_true, y_pred)

        reconstruction_loss *= output_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
        return vae_loss
    autoencoder.compile(optimizer='adam', loss=vae_loss)
    autoencoder.summary()

    # Loading model weights
    if load_weights_from_file_q.lower() == 'y':


    if more_train_q.lower() == 'y':
        # Training
        autoencoder.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=back_propagation_batch_size,
                        callbacks=[es],
                        # callbacks=[watchDog],
                        shuffle=True,
                        verbose=2,
                        validation_data=(x_test, y_test))
                # Cool down GPU
                # time.sleep(300)

    score = autoencoder.evaluate(x_test, y_test, verbose=2)
    print('Test loss of fold {}: {}'.format(fold_counter, score))
    cvscores.append(score)

    # @k evaluation process for test data
    print("eval on test data fold #{}".format(fold_counter))
    true_indices = []
    pred_indices = []
    with open(result_output_name, 'a+') as file:
        writer = csv.writer(file)
        for sample_x, sample_y in zip(x_test, y_test):
            start_time = time.time()
            sample_prediction = autoencoder.predict(np.asmatrix(sample_x))
            end_time = time.time()
            elapsed_time = (end_time - start_time)*1000
            pred_index, true_index = dblp_eval.find_indices(sample_prediction, [sample_y])
            true_indices.append(true_index[0])
            pred_indices.append(pred_index[0])
            writer.writerow([method_name, k_fold, fold_counter, len(pred_index[0][:k_max]), len(true_index[0]),
                             elapsed_time] + pred_index[0][:k_max] + true_index[0])



    # saving model
    model_json = autoencoder.to_json()

    with open('../output/Models/{}_{}_Time{}_Fold{}.json'.format(dataset_name, method_name, time_str, fold_counter), "w") as json_file:
        json_file.write(model_json)

    autoencoder.save_weights(
        "../output/Models/Weights/{}_{}_Time{}_Fold{}.h5".format(dataset_name, method_name, time_str, fold_counter))

    print('Model and its summary are saved.')

    # Deleting model from RAM
    K.clear_session()

    fold_counter += 1

print('Loss for each fold: {}'.format(cvscores))
