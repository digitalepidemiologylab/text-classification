import pytest
from dacite.exceptions import UnexpectedDataError, MissingValueError

from txtcls.utils.config_manager import *


def test_preprocess_config_folders():
    config_manager = ConfigManager(
        'test/configs/config.preprocess.folders.json', Mode.PREPROCESS)

    config = [
        PreprocessConf(
            name='preprocess_fasttext_category_merged_0',
            model=Model(name='fasttext'),
            data_init=Data(train='train.csv', test='dev.csv'),
            folders=Folders.OVERWRITE,
            path_init=Paths(
                data='data/annotation_data/train/category_merged/',
                output='./output', tmp='./tmp', other='./other'
            ),
            preprocess=Preprocess(
                standardize_func_name='standardize_fasttext_twitter',
                min_num_tokens=0,
                min_num_chars=0,
                lower_case=True,
                asciify=False,
                remove_punctuation=True,
                asciify_emoji=False,
                remove_emoji=False,
                replace_url_with=None,
                replace_user_with=None,
                replace_email_with=None,
                lemmatize=False,
                remove_stop_words=False)
        ),
        PreprocessConf(
            name='preprocess_fasttext_category_merged_1',
            model=Model(name='fasttext'),
            data_init=Data(train='train.csv', test='dev.csv'),
            folders=Folders.OVERWRITE,
            path_init=Paths(
                data='data/annotation_data/train/category_merged/',
                output='./output', tmp='./tmp', other='./other'
            ),
            preprocess=Preprocess(
                standardize_func_name='standardize_fasttext_twitter',
                min_num_tokens=0,
                min_num_chars=0,
                lower_case=True,
                asciify=False,
                remove_punctuation=False,
                asciify_emoji=False,
                remove_emoji=False,
                replace_url_with=None,
                replace_user_with=None,
                replace_email_with=None,
                lemmatize=False,
                remove_stop_words=False))
    ]

    assert config_manager.config == config


def test_preprocess_config_additional_key():
    with pytest.raises(UnexpectedDataError):
        ConfigManager(
            'test/configs/config.preprocess.additional_key.json', Mode.PREPROCESS)


def test_preprocess_config_missing_model_name():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.preprocess.missing_model_name.json',
            Mode.PREPROCESS)
        assert exc == "Please fill the 'model' key"


def test_preprocess_missing_data():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.preprocess.missing_data.json',
            Mode.PREPROCESS)
        assert exc == "Please fill the 'data' key"


def test_train_correct():
    config_manager = ConfigManager(
        'test/configs/config.train.correct.json', Mode.TRAIN)

    config = [TrainConf(
        name='train_fasttext_category_merged_1_7',
        model=TrainModel(
            name='fasttext',
            params_init='{"autotuneValidationFile": "./twitter_fasttext/'
                        'preprocess/category/preprocess_fasttext_category_'
                        'merged_7/dev.txt", "pretrainedVectors": "./twitter_'
                        'fasttext/pretrain/train_fasttext_pretrain_7/'
                        'vectors.vec", "dim": 100, "autotuneDuration": 200}',
            save_model=None,
            quantize=None,
            save_vec=None),    
        preprocess=None,
        data_init=Data(train='train.txt', test='dev.csv'),
        folders=Folders.OVERWRITE,
        path_init=Paths(
            data='./twitter_fasttext/preprocess/category/'
                 'preprocess_fasttext_category_merged_7/',
            output='./twitter_fasttext/train/category/',
            tmp='./tmp', other='./other'
        ),
        test_only=False,
        write_test_output=True)]

    assert config_manager.config == config


def test_train_missing_train_data():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.train.missing_train_data.json',
            Mode.TRAIN)
        assert exc == "Please fill the 'data.train' (and 'path.data') keys"


def test_train_missing_train_data_test_only():
    config_manager = ConfigManager(
        'test/configs/config.train.missing_train_data_test_only.json', Mode.TRAIN)

    config = [TrainConf(
        name='train_fasttext_category_merged_1_7',
        model=TrainModel(
            name='fasttext',
            params_init='{"autotuneValidationFile": "./twitter_fasttext/'
                        'preprocess/category/preprocess_fasttext_category_'
                        'merged_7/dev.txt", "pretrainedVectors": "./twitter_'
                        'fasttext/pretrain/train_fasttext_pretrain_7/'
                        'vectors.vec", "dim": 100, "autotuneDuration": 200}',
            save_model=None,
            quantize=None,
            save_vec=None),    
        preprocess=None,
        data_init=Data(test='dev.csv'),
        folders=Folders.OVERWRITE,
        path_init=Paths(
            data='./twitter_fasttext/preprocess/category/'
                 'preprocess_fasttext_category_merged_7/',
            output='./twitter_fasttext/train/category/',
            tmp='./tmp', other='./other'
        ),
        test_only=True,
        write_test_output=True)]

    assert config_manager.config == config


def test_train_missing_test_data():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.train.missing_test_data.json',
            Mode.TRAIN)
        assert exc == "Please fill the 'data.test' (and 'path.data') keys"


def train_missing_model_params():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.train.missing_model_params.json',
            Mode.PREPROCESS)
        assert exc == "Please fill the 'model.params' key"


def test_predict():
    config_manager = ConfigManager(
        'test/configs/config.predict.correct.json', Mode.PREDICT)

    config = [PredictConf(
        name='train_fasttext_category_merged_1_7',
        model=Model(name='fasttext'))]

    assert config_manager.config == config


def test_predict_missing_model_name():
    with pytest.raises(ValueError) as exc:
        ConfigManager(
            'test/configs/config.predict.missing_model_name.json',
            Mode.PREDICT)
        assert exc == "Please fill the 'model' key"


if __name__ == "__main__":
    pytest.main()
