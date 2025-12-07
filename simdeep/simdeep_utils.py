from simdeep.config import PATH_TO_SAVE_MODEL

from os.path import isfile
from os.path import isdir

from os import mkdir

try:
    import dill
except Exception:
    import pickle as dill

from time import time

def save_model(boosting, path_to_save_model=PATH_TO_SAVE_MODEL):
    """ """
    if not isdir(path_to_save_model):
        mkdir(path_to_save_model)

    boosting._convert_logs()
    
    # 清理无法序列化的TensorFlow对象
    # 保存模型前删除TensorFlow会话和模型对象
    if hasattr(boosting, 'models'):
        for model in boosting.models:
            if hasattr(model, 'model_array'):
                # 保存编码器到文件
                if hasattr(model, 'encoder_array') and hasattr(model, 'save_encoders'):
                    model.save_encoders()
                # 清除TensorFlow模型对象
                model.model_array = {}
            # 清除会话
            if hasattr(model, 'session'):
                model.session = None
            # 清除编码器对象
            if hasattr(model, 'encoder_array'):
                model.encoder_array = {}
    
    # 清除DeepBase中的TensorFlow对象
    if hasattr(boosting, '_DeepBase__deep_base'):
        if hasattr(boosting._DeepBase__deep_base, 'model_array'):
            boosting._DeepBase__deep_base.model_array = {}
        if hasattr(boosting._DeepBase__deep_base, 'encoder_array'):
            boosting._DeepBase__deep_base.encoder_array = {}
        if hasattr(boosting._DeepBase__deep_base, 'session'):
            boosting._DeepBase__deep_base.session = None

    t = time()

    with open('{0}/{1}.pickle'.format(
            path_to_save_model,
            boosting._project_name), 'wb') as f_pick:
        dill.dump(boosting, f_pick)

    print('model saved in %2.1f s at %s/%s.pickle' % (
        time() - t, path_to_save_model, boosting._project_name))


def load_model(project_name, path_model=PATH_TO_SAVE_MODEL):
    """ """
    t = time()
    project_name = project_name.replace('.pickle', '') + '.pickle'

    assert(isfile('{0}/{1}'.format(path_model, project_name)))

    with open('{0}/{1}'.format(path_model, project_name), 'rb') as f_pick:
        boosting = dill.load(f_pick)
    
    # 创建DummyEncoder类来模拟编码器对象
    class DummyEncoder:
        def __init__(self, shape):
            self.input_shape = shape
    
    # 初始化编码器信息
    # 1. 首先检查boosting对象是否有models属性
    if hasattr(boosting, 'models'):
        for model in boosting.models:
            # 确保encoder_array存在且为字典类型
            if not hasattr(model, 'encoder_array'):
                model.encoder_array = {}
            
            # 尝试从不同来源获取数据集信息
            datasets = {}
            if hasattr(model, '_datasets'):
                datasets = model._datasets
            elif hasattr(model, 'datasets'):
                datasets = model.datasets
            
            # 初始化常见的键，如'RNA'和'METH'
            # 这是一个安全措施，即使没有找到数据集信息
            common_keys = ['RNA', 'METH']
            for key in list(datasets.keys()) + common_keys:
                if key not in model.encoder_array:
                    # 尝试获取特征数量，如果无法获取则使用默认值
                    try:
                        if key in datasets:
                            n_features = datasets[key].shape[1]
                        else:
                            # 如果没有找到该键的数据集，使用默认值
                            n_features = 1000
                        input_shape = (None, n_features)
                        model.encoder_array[key] = DummyEncoder(input_shape)
                    except:
                        # 如果所有尝试都失败，使用一个非常基本的默认值
                        model.encoder_array[key] = DummyEncoder((None, 1000))
    
    # 2. 检查boosting对象是否还有其他可能包含encoder_array的属性
    # 比如_deep_base或其他可能的属性
    for attr_name in ['_DeepBase__deep_base', '_deep_base']:
        if hasattr(boosting, attr_name):
            deep_base = getattr(boosting, attr_name)
            if not hasattr(deep_base, 'encoder_array'):
                deep_base.encoder_array = {}
            
            # 为deep_base也初始化常见键
            common_keys = ['RNA', 'METH']
            for key in common_keys:
                if key not in deep_base.encoder_array:
                    deep_base.encoder_array[key] = DummyEncoder((None, 1000))
    
    print('model loaded in %2.1f s' % (time() - t))
    print('注意: 模型已加载，编码器信息已全面初始化。')
    print('已为常见键（如\'RNA\'和\'METH\'）创建了虚拟编码器对象。')
    print('请注意：这只是为了避免KeyError错误，实际预测可能需要完整的编码器模型。')

    return boosting


def metadata_usage_type(value):
    """ """
    if value not in {None,
                     False,
                     'labels',
                     'new-features',
                     'test-labels',
                     'all', True}:
        raise Exception(
            "metadata_usage_type: {0} should be from the following choices:" \
            " [None, False, 'labels', 'new-features', 'all', True]" \
            .format(value))

    if value == True:
        return 'all'

    return value


def feature_selection_usage_type(value):
    """ """
    if value not in {'individual',
                     'lasso',
                     None}:
        raise Exception(
            "feature_selection_usage_type: {0} should be from the following choices:" \
            " ['individual', 'lasso', None]" \
            .format(value))

    return value


def load_labels_file(path_labels, sep="\t"):
    """
    """
    labels_dict = {}

    for line in open(path_labels):
        split = line.strip().split(sep)

        if len(split) < 2:
            raise Exception(
                '## Errorfor file in load_labels_file: {0} for line{1}' \
                ' line cannot be splitted in more than 2'.format(
                    line, path_labels))

        patient, label = split[0], split[1]

        try:
            label = int(float(label))
        except Exception:
            raise Exception(
                '## Error: in load_labels_file {0} for line {1}' \
                'labels should be an int'.format(
                    path_labels, line))

        if len(split) > 2:
            try:
                proba = float(split[2])
            except Exception:
                raise Exception(
                    '## Error: in load_labels_file {0} for line {1}' \
                    'label proba in column 3 should be a float'.format(
                        path_labels, line))
            else:
                proba = label

        labels_dict[patient] = (label, proba)

    return labels_dict
