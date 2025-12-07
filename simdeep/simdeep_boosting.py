import warnings
from simdeep.simdeep_analysis import SimDeep
from simdeep.extract_data import LoadData

from simdeep.coxph_from_r import coxph
from simdeep.coxph_from_r import c_index
from simdeep.coxph_from_r import c_index_multiple

from sklearn.model_selection import KFold
# from sklearn.preprocessing import OneHotEncoder

from collections import Counter
from collections import defaultdict
from itertools import combinations

import numpy as np

from scipy.stats import gmean
from sklearn.metrics import adjusted_rand_score

from simdeep.config import PROJECT_NAME
from simdeep.config import PATH_RESULTS
from simdeep.config import NB_THREADS
from simdeep.config import NB_ITER
from simdeep.config import NB_FOLDS
from simdeep.config import CLASS_SELECTION
from simdeep.config import NB_CLUSTERS
from simdeep.config import NORMALIZATION
from simdeep.config import EPOCHS
from simdeep.config import NEW_DIM
from simdeep.config import NB_SELECTED_FEATURES
from simdeep.config import PVALUE_THRESHOLD
from simdeep.config import CLUSTER_METHOD
from simdeep.config import CLASSIFICATION_METHOD
from simdeep.config import TRAINING_TSV
from simdeep.config import SURVIVAL_TSV
from simdeep.config import PATH_DATA
from simdeep.config import SURVIVAL_FLAG
from simdeep.config import NODES_SELECTION
from simdeep.config import CINDEX_THRESHOLD
from simdeep.config import USE_AUTOENCODERS
from simdeep.config import FEATURE_SURV_ANALYSIS
from simdeep.config import CLUSTERING_OMICS
from simdeep.config import USE_R_PACKAGES_FOR_SURVIVAL

# Parameter for autoencoder
from simdeep.config import LEVEL_DIMS_IN
from simdeep.config import LEVEL_DIMS_OUT
from simdeep.config import LOSS
from simdeep.config import OPTIMIZER
from simdeep.config import ACT_REG
from simdeep.config import W_REG
from simdeep.config import DROPOUT
from simdeep.config import ACTIVATION
from simdeep.config import PATH_TO_SAVE_MODEL
from simdeep.config import DATA_SPLIT
from simdeep.config import MODEL_THRES

from multiprocessing import Pool

from simdeep.deepmodel_base import DeepBase

try:
    import simplejson as json
except Exception:
    import json

from distutils.dir_util import mkpath

from os.path import isdir
from os import mkdir

from glob import glob

import gc

from time import time

from numpy import hstack
from numpy import vstack

import pandas as pd

from simdeep.survival_utils import \
    _process_parallel_feature_importance_per_cluster
from simdeep.survival_utils import \
    _process_parallel_survival_feature_importance_per_cluster



class SimDeepBoosting():
    """
    Instanciate a new DeepProg Boosting instance.
    The default parameters are defined in the config.py file

    Parameters:
            :nb_it: Number of models to construct
            :do_KM_plot: Plot Kaplan-Meier (default: True)
            :distribute: Distribute DeepProg using ray (default:  False)
            :nb_threads: Number of python threads to use to compute parallel Cox-PH
            :class_selection: Consensus score to agglomerate DeepProg Instance {'mean', 'max', 'weighted_mean', 'weighted_max'} (default: 'mean')
            :model_thres: Cox-PH p-value threshold to reject a model for DeepProg Boosting module
            :verbose: Verobosity (Default: True)
            :seed: Seed defining the  random split of the training dataset (Default: None).
            :project_name: Project name used to save files
            :use_autoencoders: Use autoencoder steps to embed the data (default: True)
            :feature_surv_analysis: Use individual survival feature detection to filter out features (default: True)
            :split_n_fold: For each instance, the original dataset is split in folds and one fold is left
            :path_results: Path to create a result folder
            :nb_clusters: Number of clusters to use
            :epochs: Number of epochs
            :normalization: Normalisation procedure to use. See config.py file for details
            :nb_selected_features: Number of top features selected for classification
            :cluster_method: Clustering method. possible choice: ['mixture', 'kmeans', 'coxPH'] or class instance having fit and fit_proba attributes
            :pvalue_thres: Threshold for survival significance to set a node as valid
            :classification_method: Possible choice: {'ALL_FEATURES', 'SURVIVAL_FEATURES'} (default: 'ALL_FEATURES')
            :new_dim: Size of the new embedding
            :training_tsv: Input matrix files
            :survival_tsv: Input surival file
            :survival_flag: Survival flag to use
            :path_data: Path of the input file
            :level_dims_in: Autoencoder node layers before the middle layer (default: [])
            :level_dims_out: Autoencoder node layers after the middle layer (default: [])
            :loss: Loss function to minimize (default: 'binary_crossentropy')
            :optimizer: Optimizer (default: adam)
            :act_reg: L2 Regularization constant on the node activity (default: False)
            :w_reg: L1 Regularization constant on the weight (default: False)
            :dropout: Percentage of edges being dropout at each training iteration (None for no dropout) (default: 0.5)
            :data_split: Fraction of the dataset to be used as test set when building the autoencoder (default: None)
            :node_selection: possible choice: {'Cox-PH', 'C-index'} (default: Cox-PH)
            :cindex_thres: Valid if 'c-index' is chosen (default: 0.65)
            :activation: Activation function (default: 'tanh')
            :clustering_omics: Which omics to use for clustering. If empty, then all the available omics will be used (default [] => all)
            :path_to_save_model: path to save the model
            :metadata_usage: Meta data usage with survival models (if metadata_tsv provided as argument to the dataset). Possible choice are [None, False, 'labels', 'new-features', 'all', True] (True is the same as all)
            :subset_training_with_meta: Use a metadata key-value dict {meta_key:value} to subset the training sets
            :alternative_embedding: alternative external embedding to use instead of building autoencoders (default None)
            :kwargs_alternative_embedding: parameters for external embedding fitting
    """
    def __init__(self,
                 nb_it=NB_ITER,
                 do_KM_plot=True,
                 distribute=False,
                 nb_threads=NB_THREADS,
                 class_selection=CLASS_SELECTION,
                 model_thres=MODEL_THRES,
                 verbose=True,
                 seed=None,
                 project_name='{0}_boosting'.format(PROJECT_NAME),
                 use_autoencoders=USE_AUTOENCODERS,
                 feature_surv_analysis=FEATURE_SURV_ANALYSIS,
                 split_n_fold=NB_FOLDS,
                 path_results=PATH_RESULTS,
                 nb_clusters=NB_CLUSTERS,
                 epochs=EPOCHS,
                 normalization=NORMALIZATION,
                 nb_selected_features=NB_SELECTED_FEATURES,
                 cluster_method=CLUSTER_METHOD,
                 pvalue_thres=PVALUE_THRESHOLD,
                 classification_method=CLASSIFICATION_METHOD,
                 new_dim=NEW_DIM,
                 training_tsv=TRAINING_TSV,
                 metadata_usage=None,
                 survival_tsv=SURVIVAL_TSV,
                 metadata_tsv=None,
                 subset_training_with_meta={},
                 survival_flag=SURVIVAL_FLAG,
                 path_data=PATH_DATA,
                 level_dims_in=LEVEL_DIMS_IN,
                 level_dims_out=LEVEL_DIMS_OUT,
                 loss=LOSS,
                 optimizer=OPTIMIZER,
                 act_reg=ACT_REG,
                 w_reg=W_REG,
                 dropout=DROPOUT,
                 data_split=DATA_SPLIT,
                 node_selection=NODES_SELECTION,
                 cindex_thres=CINDEX_THRESHOLD,
                 activation=ACTIVATION,
                 clustering_omics=CLUSTERING_OMICS,
                 path_to_save_model=PATH_TO_SAVE_MODEL,
                 feature_selection_usage='individual',
                 use_r_packages=USE_R_PACKAGES_FOR_SURVIVAL,
                 alternative_embedding=None,
                 kwargs_alternative_embedding={},
                 **additional_dataset_args):
        ''' '''
        assert(class_selection in ['max', 'mean', 'weighted_mean', 'weighted_max'])
        self.class_selection = class_selection

        self._instance_weights = None
        self.distribute = distribute
        self.model_thres = model_thres
        self.models = []
        self.verbose = verbose
        self.nb_threads = nb_threads
        self.nb_it = nb_it
        self.do_KM_plot = do_KM_plot
        self.project_name = project_name
        self._project_name = project_name
        self.path_results = '{0}/{1}'.format(path_results, project_name)
        self.training_tsv = training_tsv
        self.survival_tsv = survival_tsv
        self.survival_flag = survival_flag
        self.path_data = path_data
        self.dataset = None
        self.cindex_thres = cindex_thres
        self.node_selection = node_selection
        self.clustering_omics = clustering_omics
        self.metadata_tsv = metadata_tsv
        self.metadata_usage = metadata_usage
        self.feature_selection_usage = feature_selection_usage
        self.subset_training_with_meta = subset_training_with_meta
        self.use_r_packages = use_r_packages

        self.metadata_mat_full = None

        self.cluster_method = cluster_method
        self.use_autoencoders = use_autoencoders
        self.feature_surv_analysis = feature_surv_analysis

        if self.feature_selection_usage is None:
            self.feature_surv_analysis = False

        self.encoder_for_kde_plot_dict = {}
        self.kde_survival_node_ids = {}
        self.kde_train_matrices = {}

        if not isdir(self.path_results):
            try:
                mkpath(self.path_results)
            except Exception:
                print('cannot find or create the current result path: {0}' \
                      '\n consider changing it as option' \
                      .format(self.path_results))

        self.test_tsv_dict = None
        self.test_survival_file = None
        self.test_normalization = None

        self.test_labels = None
        self.test_labels_proba = None

        self.cv_labels = None
        self.cv_labels_proba = None
        self.full_labels = None
        self.full_labels_dicts = None
        self.full_labels_proba = None
        self.survival_full = None
        self.sample_ids_full = None
        self.feature_scores_per_cluster = {}
        self.survival_feature_scores_per_cluster = {}

        self._pretrained_fit = False

        self.log = {}

        self.alternative_embedding = alternative_embedding
        self.kwargs_alternative_embedding = kwargs_alternative_embedding

        ######## deepprog instance parameters ########
        self.nb_clusters = nb_clusters
        self.normalization = normalization
        self.epochs = epochs
        self.new_dim = new_dim
        self.nb_selected_features = nb_selected_features
        self.pvalue_thres = pvalue_thres
        self.cluster_method = cluster_method
        self.cindex_test_folds = []
        self.classification_method = classification_method
        ##############################################

        self.test_fname_key = ''
        self.matrix_with_cv_array = None

        autoencoder_parameters = {
            'epochs': self.epochs,
            'new_dim': self.new_dim,
            'level_dims_in': level_dims_in,
            'level_dims_out': level_dims_out,
            'loss': loss,
            'optimizer': optimizer,
            'act_reg': act_reg,
            'w_reg': w_reg,
            'dropout': dropout,
            'data_split': data_split,
            'activation': activation,
            'path_to_save_model': path_to_save_model,
        }

        self.datasets = []
        self.seed = seed

        self.log['parameters'] = {}

        for arg in self.__dict__:
            self.log['parameters'][arg] = str(self.__dict__[arg])

        self.log['seed'] = seed
        self.log['parameters'].update(autoencoder_parameters)

        self.log['nb_it'] = nb_it
        self.log['normalization'] = normalization
        self.log['nb clusters'] = nb_clusters
        self.log['success'] = False
        self.log['survival_tsv'] = self.survival_tsv
        self.log['metadata_tsv'] = self.metadata_tsv
        self.log['subset_training_with_meta'] = self.subset_training_with_meta
        self.log['training_tsv'] = self.training_tsv
        self.log['path_data'] = self.path_data

        additional_dataset_args['survival_tsv'] = self.survival_tsv
        additional_dataset_args['metadata_tsv'] = self.metadata_tsv
        additional_dataset_args['subset_training_with_meta'] = self.subset_training_with_meta
        additional_dataset_args['training_tsv'] = self.training_tsv
        additional_dataset_args['path_data'] = self.path_data
        additional_dataset_args['survival_flag'] = self.survival_flag

        if 'fill_unkown_feature_with_0' in additional_dataset_args:
            self.log['fill_unkown_feature_with_0'] = additional_dataset_args[
                'fill_unkown_feature_with_0']

        self.ray = None

        self._init_datasets(nb_it, split_n_fold,
                            autoencoder_parameters,
                            **additional_dataset_args)

    def _init_datasets(self, nb_it, split_n_fold,
                       autoencoder_parameters,
                       **additional_dataset_args):
        """
        """
        if self.seed:
            np.random.seed(self.seed)
        else:
            self.seed = np.random.randint(0, 10000000)

        max_seed = 1000
        min_seed = 0

        if self.seed > max_seed:
            min_seed = self.seed - max_seed
            max_seed = self.seed

        # 使用确定性方式生成random_states，确保可复现
        # 使用固定的种子序列，确保每次运行生成相同的random_states
        np.random.seed(self.seed)
        # 为每个迭代生成不同的但确定性的随机种子
        random_states = []
        for it in range(nb_it):
            # 使用确定性的种子生成方式
            iter_seed = self.seed * 1000 + it
            np.random.seed(iter_seed)
            random_states.append(np.random.randint(min_seed, max_seed))
        random_states = np.array(random_states)

        self.split_n_fold = split_n_fold

        for it in range(nb_it):
            if self.split_n_fold:
                split = KFold(n_splits=split_n_fold,
                              shuffle=True, random_state=random_states[it])
            else:
                split = None

            autoencoder_parameters['seed'] = random_states[it]

            dataset = LoadData(cross_validation_instance=split,
                               verbose=False,
                               normalization=self.normalization,
                               _autoencoder_parameters=autoencoder_parameters.copy(),
                               **additional_dataset_args)

            self.datasets.append(dataset)

    def __del__(self):
        """
        """
        for model in self.models:
            del model

        try:
            gc.collect()
        except Exception as e:
            print('Warning: Exception {0} from garbage collector. continuing... '.format(
                e))

    def _from_models(self, fname, *args, **kwargs):
        """
        """
        if self.distribute:
            return self.ray.get([getattr(model, fname).remote(*args, **kwargs)
                                 for model in self.models])
        else:
            return [getattr(model, fname)(*args, **kwargs)
                    for model in self.models]


    def _from_model(self, model, fname, *args, **kwargs):
        """
        """
        if self.distribute:
            return self.ray.get(getattr(model, fname).remote(
                *args, **kwargs))
        else:
            return getattr(model, fname)(*args, **kwargs)

    def _from_model_attr(self, model, atname):
        """
        """
        if self.distribute:
            return self.ray.get(model._get_attibute.remote(atname))
        else:
            return model._get_attibute(atname)

    def _from_models_attr(self, atname):
        """
        """
        if self.distribute:
            return self.ray.get([model._get_attibute.remote(atname)
                                 for model in self.models])
        else:
            return [model._get_attibute(atname) for model in self.models]

    def _from_model_dataset(self, model, atname):
        """
        """
        if self.distribute:
            return self.ray.get(model._get_from_dataset.remote(atname))
        else:
            return model._get_from_dataset(atname)


    def _do_class_selection(self, inputs, **kwargs):
        """
        """
        if self.class_selection == 'max':
            return  _highest_proba(inputs)
        elif self.class_selection == 'mean':
            return _mean_proba(inputs)
        elif self.class_selection == 'weighted_mean':
            return _weighted_mean(inputs, **kwargs)
        elif self.class_selection == 'weighted_max':
            return _weighted_max(inputs, **kwargs)

    def partial_fit(self, debug=False):
        """
        """
        self._fit(debug=debug)

    def fit_on_pretrained_label_file(
            self,
            labels_files=[],
            labels_files_folder="",
            file_name_regex="*.tsv",
            verbose=False,
            debug=False,
    ):
        """
        fit a deepprog simdeep models without training autoencoders but using isntead  ID->labels files (one for each model instance)
        """
        assert(isinstance((labels_files), list))

        if not labels_files and not labels_files_folder:
            raise Exception(
                '## Error with fit_on_pretrained_label_file: ' \
                ' either labels_files or labels_files_folder should be non empty')

        if not labels_files:
            labels_files = glob('{0}/{1}'.format(labels_files_folder,
                                                 file_name_regex))

        if not labels_files:
            raise Exception('## Error: labels_files empty')

        self.fit(
            verbose=verbose,
            debug=debug,
            pretrained_labels_files=labels_files)

    def fit(self, debug=False, verbose=False, pretrained_labels_files=[]):
        """
        if pretrained_labels_files, is given, the models are constructed using these labels
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if pretrained_labels_files:
                self._pretrained_fit = True
            else:
                self._pretrained_fit = False

            if self.distribute:
                self._fit_distributed(
                    pretrained_labels_files=pretrained_labels_files)
            else:
                self._fit(
                    debug=debug,
                    verbose=verbose,
                    pretrained_labels_files=pretrained_labels_files)

    def _fit(self, debug=False, verbose=False, pretrained_labels_files=[]):
        ''' '''
        print('fit models...')
        start_time = time()

        try:
            self.models = [SimDeep(
                nb_clusters=self.nb_clusters,
                nb_selected_features=self.nb_selected_features,
                pvalue_thres=self.pvalue_thres,
                dataset=self.datasets[i],
                load_existing_models=False,
                verbose=self.datasets[i].verbose,
                _isboosting=True,
                do_KM_plot=False,
                cluster_method=self.cluster_method,
                clustering_omics=self.clustering_omics,
                use_autoencoders=self.use_autoencoders,
                use_r_packages=self.use_r_packages,
                feature_surv_analysis=self.feature_surv_analysis,
                path_results=self.path_results,
                project_name=self.project_name,
                classification_method=self.classification_method,
                cindex_thres=self.cindex_thres,
                alternative_embedding=self.alternative_embedding,
                kwargs_alternative_embedding=self.kwargs_alternative_embedding,
                node_selection=self.node_selection,
                metadata_usage=self.metadata_usage,
                feature_selection_usage=self.feature_selection_usage
            ) for i in range(self.nb_it)]
        except Exception as e:
            print(f"An error occurred during model initialization: {e}")
            self.models = []
        for m in self.models:
            try:
                m.load_training_dataset()
                m.fit()
            except Exception as e:
                print('model with random state:{1} didn\'t converge:{0}'.format(str(e), m.seed))

    def _aggregate_proba(self, probas_list):
        if not probas_list:
            return None
        max_k = max(p.shape[1] for p in probas_list)
        adjusted = []
        for p in probas_list:
            if p.shape[1] < max_k:
                pad = np.zeros((p.shape[0], max_k - p.shape[1]))
                p = np.hstack([p, pad])
            adjusted.append(p)
        return sum(adjusted) / len(adjusted)

    def _aggregate_training_labels(self):
        probas = self._from_models_attr('labels_proba')
        if not probas:
            return
        agg = self._aggregate_proba(probas)
        self.labels_proba = agg
        self.labels = np.argmax(agg, axis=1)

    def predict_labels_on_full_dataset(self):
        if not self.models:
            raise Exception('no models')
        _ = self._from_models('predict_labels_on_full_dataset')
        probas = self._from_models_attr('full_labels_proba')
        self.survival_full = self._from_model_dataset(self.models[0], 'survival_full')
        self.sample_ids_full = self._from_model_dataset(self.models[0], 'sample_ids_full')
        agg = self._aggregate_proba(probas)
        self.full_labels_proba = agg
        self.full_labels = np.argmax(agg, axis=1)
        nbdays, isdead = np.asarray(self.survival_full).T.tolist()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        df = pd.DataFrame({'labels': self.full_labels, 'nbdays': nbdays, 'isdead': isdead})
        pvalue = coxph(
            df, time_col='nbdays', event_col='isdead', features=['labels'],
            isfactor=False, seed=self.seed, use_r_packages=self.use_r_packages,
            do_KM_plot=self.do_KM_plot, png_path=self.path_results,
            fig_name='{0}_full_labels_KM_plot_boosting_full_{1}_seed{2}'.format(self.project_name, timestamp, self.seed),
            plot_title='full / labels / seed {0}'.format(self.seed)
        )

        dfp = pd.DataFrame({'labels_proba': self.full_labels_proba.T[0], 'nbdays': nbdays, 'isdead': isdead})
        pvalue_proba = coxph(
            dfp, time_col='nbdays', event_col='isdead', features=['labels_proba'],
            isfactor=False, seed=self.seed, use_r_packages=self.use_r_packages,
            do_KM_plot=self.do_KM_plot, png_path=self.path_results,
            fig_name='{0}_full_proba_KM_plot_boosting_full_{1}_seed{2}'.format(self.project_name, timestamp, self.seed),
            plot_title='full / proba / seed {0}'.format(self.seed)
        )
        self.full_pvalue = pvalue
        self.full_pvalue_proba = pvalue_proba
        return pvalue, pvalue_proba

    def compute_c_indexes_for_full_dataset(self):
        if not hasattr(self, 'full_labels_proba'):
            return np.nan
        days_full, dead_full = np.asarray(self.survival_full).T.tolist()
        act = self.full_labels_proba.T[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cindex = c_index(act, dead_full, days_full)
        except Exception:
            cindex = np.nan
        return cindex

    def load_new_test_dataset(self, tsv_dict, fname_key='', path_survival_file=None, normalization=None, survival_flag=None, metadata_file=None):
        self.test_fname_key = fname_key
        for m in self.models:
            self._from_model(m, 'load_new_test_dataset', tsv_dict, fname_key, path_survival_file, normalization, survival_flag=survival_flag, metadata_file=metadata_file)

    def predict_labels_on_test_dataset(self):
        if not self.models:
            raise Exception('no models')
        _ = self._from_models('predict_labels_on_test_dataset')
        probas = self._from_models_attr('test_labels_proba')
        self.survival_test = self._from_model_dataset(self.models[0], 'survival_test')
        self.sample_ids_test = self._from_model_dataset(self.models[0], 'sample_ids_test')
        agg = self._aggregate_proba(probas)
        self.test_labels_proba = agg
        self.test_labels = np.argmax(agg, axis=1)
        nbdays, isdead = np.asarray(self.survival_test).T.tolist()
        
        # Generate timestamp for filenames
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        df = pd.DataFrame({'labels': self.test_labels, 'nbdays': nbdays, 'isdead': isdead})
        pvalue = coxph(df, time_col='nbdays', event_col='isdead', features=['labels'], 
                       isfactor=False, 
                       seed=self.seed, 
                       use_r_packages=self.use_r_packages,
                       do_KM_plot=self.do_KM_plot,
                       png_path=self.path_results,
                       fig_name='{0}_val_labels_KM_plot_boosting_{1}_{2}_seed{3}'.format(self.project_name, self.test_fname_key, timestamp, self.seed),
                       plot_title='{0} / labels / seed {1}'.format(self.test_fname_key, self.seed))
                       
        dfp = pd.DataFrame({'labels_proba': self.test_labels_proba.T[0], 'nbdays': nbdays, 'isdead': isdead})
        pvalue_proba = coxph(
            dfp,
            time_col='nbdays',
            event_col='isdead',
            features=['labels_proba'],
            isfactor=False,
            seed=self.seed,
            use_r_packages=self.use_r_packages,
            do_KM_plot=self.do_KM_plot,
            png_path=self.path_results,
            fig_name='{0}_val_proba_KM_plot_boosting_{1}_{2}_seed{3}'.format(
                self.project_name, self.test_fname_key, timestamp, self.seed
            ),
            plot_title='{0} / proba / seed {1}'.format(self.test_fname_key, self.seed)
        )
                             
        self.test_pvalue = pvalue
        self.test_pvalue_proba = pvalue_proba
        return pvalue, pvalue_proba

    def compute_c_indexes_for_test_dataset(self):
        if not hasattr(self, 'test_labels_proba'):
            return np.nan
        days_test, dead_test = np.asarray(self.survival_test).T.tolist()
        act = self.test_labels_proba.T[0]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cindex = c_index(act, dead_test, days_test)
        except Exception:
            cindex = np.nan
        return cindex

    def _fit_distributed(self, pretrained_labels_files=[]):
        ''' '''
        print('fit models...')
        start_time = time()

        from simdeep.simdeep_distributed import SimDeepDistributed
        import ray
        assert(ray.is_initialized())
        self.ray = ray

        try:
            self.models = [SimDeepDistributed.remote(
                nb_clusters=self.nb_clusters,
                nb_selected_features=self.nb_selected_features,
                pvalue_thres=self.pvalue_thres,
                dataset=self.datasets[i],
                load_existing_models=False,
                verbose=self.datasets[i].verbose,
                _isboosting=True,
                do_KM_plot=False,
                cluster_method=self.cluster_method,
                clustering_omics=self.clustering_omics,
                use_autoencoders=self.use_autoencoders,
                use_r_packages=self.use_r_packages,
                feature_surv_analysis=self.feature_surv_analysis,
                path_results=self.path_results,
                project_name=self.project_name,
                classification_method=self.classification_method,
                cindex_thres=self.cindex_thres,
                alternative_embedding=self.alternative_embedding,
                kwargs_alternative_embedding=self.kwargs_alternative_embedding,
                node_selection=self.node_selection,
                metadata_usage=self.metadata_usage,
                feature_selection_usage=self.feature_selection_usage
            ) for i in range(self.nb_it)]
        except Exception as e:
            print(f"An error occurred during model initialization: {e}")
            self.models = []
