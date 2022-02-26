import logging
from typing import List
import matplotlib.pyplot as plt
from keras import backend as K
import keras_tuner as kt
import tfx.v1 as tfx
import keras_tuner
import tensorflow as tf
import tensorflow_transform as tft
import pprint
# Features with string data types that will be converted to indices
from tfx.components.trainer.fn_args_utils import DataAccessor, FnArgs
from tfx_bsl.tfxio import dataset_options

CATEGORICAL_FEATURE_KEYS = [
    'product_age_group', 'product_id', 'product_title', 'user_id', 'device_type', 'audience_id', 'product_gender',
    'product_brand', 'product_category(1)', 'product_category(2)', 'product_category(3)', 'product_category(4)',
    'product_category(5)', 'product_category(6)', 'product_category(7)', 'product_country', 'partner_id'
]

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['nb_clicks_1week']

# Feature that can be grouped into buckets


# Feature that the model will predict
TIME_KEY = 'time_delay_for_conversion'
SALE = 'Sale'


# Utility function for renaming the feature
def transformed_name(key):
    key = key.replace('(', '')
    key = key.replace(')', '')
    return key


_NUMERIC_FEATURE_KEYS = NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = CATEGORICAL_FEATURE_KEYS
_SALE = SALE
_transformed_name = transformed_name


# Define the transformations
def preprocessing_fn(inputs):

    outputs = {}

    # Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(_fill_in_missing(inputs[key]))

    # for key in _SALE:
    #     print(key)

    #     pprint(_TIME_KEY)
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            num_oov_buckets=1,
            vocab_filename=key)
    # outputs[_transformed_name('time_delay_for_conversion')] = inputs['time_delay_for_conversion']
    outputs['Sale'] = _fill_in_missing(inputs['Sale'])
    # Convert the label strings to an index

    return outputs


_DENSE_FLOAT_FEATURE_KEYS = NUMERIC_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = CATEGORICAL_FEATURE_KEYS
_VOCAB_SIZE = 40
_VOCAB_SIZE = 1000
_transformed_name = transformed_name

# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
_OOV_SIZE = 10
_LABEL_KEY = SALE


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.
    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
            default_value),
        axis=1)


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    # We need to track the layers in the model in order to save it.
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(_LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)

        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _gzip_reader_fn(filenames):
    '''Load compressed dataset

    '''
    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def input_gen(file_pattern,
              tf_transform_output,
              num_epochs=None,
              is_train=True,
              batch_size=32) -> tf.data.Dataset:
    '''Create batches of features and labels from TF Records

    Args:
    file_pattern - files pathes for dataset
    tf_transform_output - transform output graph
    num_epochs -
    batch_size -

    Returns:
    dataset
    '''
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())
    print('salammmm ', transformed_feature_spec)
    if is_train:
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=_gzip_reader_fn,
            num_epochs=num_epochs,
            label_key=_LABEL_KEY)
    else:
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=file_pattern,
            batch_size=batch_size,
            features=transformed_feature_spec,
            reader=_gzip_reader_fn,
            num_epochs=num_epochs,
        )

    return dataset


# def _input_fn(file_pattern: List[str],
#               data_accessor: DataAccessor,
#               tf_transform_output: tft.TFTransformOutput,
#               batch_size: int = 200) -> tf.data.Dataset:
#     """Generates features and label for tuning/training.
#     Args:
#       file_pattern: List of paths or patterns of input tfrecord files.
#       data_accessor: DataAccessor for converting input to RecordBatch.
#       tf_transform_output: A TFTransformOutput.
#       batch_size: representing the number of consecutive elements of returned
#         dataset to combine in a single batch
#     Returns:
#       A dataset that contains (features, indices) tuple where features is a
#         dictionary of Tensors, and indices is a single Tensor of label indices.
#     """
#     return data_accessor.tf_dataset_factory(
#         file_pattern,
#         dataset_options.TensorFlowDatasetOptions(
#             batch_size=batch_size, label_key=(_LABEL_KEY)),
#         tf_transform_output.transformed_metadata.schema).repeat()

#not used in current version - just for base line
def _build_keras_model_soft(hparams: keras_tuner.HyperParameters, base_model=None):
    inputs = [tf.keras.layers.Input(shape=(1,), name=f) for f in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)]
    inputs += [tf.keras.layers.Input(shape=(1,), name=f) for f in _transformed_names(_CATEGORICAL_FEATURE_KEYS)]
    d = tf.keras.layers.concatenate(inputs)
    d = tf.keras.layers.Dense(8, activation='relu')(d)
    d = tf.keras.layers.Dense(8, activation='relu')(d)
    outputs = tf.keras.layers.Dense(
        1, activation='sigmoid')(d)
    outputs = tf.squeeze(outputs, -1)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-2),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.FalseNegatives()])
    print(model.summary())

    return model


def _build_keras_model(hparams: keras_tuner.HyperParameters, base_model=None) -> tf.keras.Model:
    """Creates a DNN Keras model for classifying  data.
  Args:
    hidden_units: [int], the layer sizes of the DNN (input layer first).
  Returns:
    A keras Model.
  """

    real_valued_columns = [
        tf.feature_column.numeric_column(key, shape=())
        for key in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            key, num_buckets=_VOCAB_SIZE + 1, default_value=0)
        for key in _transformed_names(_VOCAB_FEATURE_KEYS)
    ]

    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]

    model = _wide_and_deep_classifier(
        wide_columns=real_valued_columns,
        deep_columns=indicator_column,
        dnn_hidden_units=int(hparams.get('num_layers')) or [20, 15, 10, 5], lr=0.001, base_model=base_model)
    return model


def _wide_and_deep_classifier(wide_columns, deep_columns, dnn_hidden_units, lr, base_model=None):
    """Build a simple keras wide and deep model.
     Returns:
     A Wide and Deep Keras model
   """
    # Following values are hard coded for simplicity in this example,
    # However prefarably they should be passsed in as hparams.

    # Keras needs the feature definitions at compile time.
    # TODO: Automate generation of input layers from FeatureColumn.

    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in (_DENSE_FLOAT_FEATURE_KEYS)
    }

    input_layers.update({
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype='int32')
        for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
    })

    # TODO(b/161952382): Replace with Keras premade models and
    # Keras preprocessing layers.
    if True:
        if base_model:
            base_model.trainable = False
            inp = base_model.input  # input placeholder
            outputs = [base_model.layers[-4].output]  # layer outputs
            functors = [K.function([inp], [out]) for out in outputs]
            #             deep = functors[0](input_layers)[0]

            deep = [func(input_layers) for func in functors][0][0]
            deep = tf.keras.layers.BatchNormalization()(deep)
            deep = tf.keras.layers.ReLU()(deep)
            print("salam", deep)
        else:
            print("sajlam")
            deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
            for numnodes in [32, 16]:
                deep = tf.keras.layers.Dense(numnodes)(deep)
                deep = tf.keras.layers.Dropout(0.3)(deep)

        wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)

        output = tf.keras.layers.Dense(
            1, activation='sigmoid')(
            tf.keras.layers.concatenate([deep, wide]))
        output = tf.squeeze(output, -1)
    else:
        wide = tf.keras.layers.BatchNormalization()(input_layers)
        deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
        for units in [64, 32, 16]:
            deep = tf.keras.layers.Dense(units)(deep)
            deep = tf.keras.layers.BatchNormalization()(deep)
            deep = tf.keras.layers.ReLU()(deep)
            deep = tf.keras.layers.Dropout(0.3)(deep)
        merged = tf.keras.layers.concatenate([wide, deep])
        output = tf.keras.layers.Dense(units=2, activation="softmax")(merged)

    model = tf.keras.Model(input_layers, output)
    print(model.layers)
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.FalseNegatives()])
    model.summary(print_fn=logging.info)
    print(model.summary())
    return model


def create_model_inputs():
    inputs = {}
    for feature_name in _DENSE_FLOAT_FEATURE_KEYS:
        inputs[feature_name] = tf.keras.layers.Input(
            name=feature_name, shape=(), dtype=tf.float32
        )
    for feature_name in CATEGORICAL_FEATURE_KEYS:
        inputs[feature_name] = tf.keras.layers.Input(
            name=feature_name, shape=(), dtype=tf.string
        )
    return inputs


def _get_hyperparameters() -> keras_tuner.HyperParameters:
    hp = keras_tuner.HyperParameters()
    # Defines search space.
    hp.Choice('learning_rate', [1e-2, 1e-3], default=1e-2)
    hp.Int('num_layers', 3, 6, default=4)
    return hp


def tuner_fn(fn_args: FnArgs) -> tfx.components.TunerFnResult:
    """Build the tuner using the KerasTuner API.
    """
    # BaseTuner.
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    tuner = kt.BayesianOptimization(
        _build_keras_model,

        objective=kt.Objective("val_loss", direction="min"),
        max_trials=12,
        num_initial_points=5,
        alpha=0.0001,
        beta=2.6,
        seed=None,
        hyperparameters=_get_hyperparameters(),
        tune_new_entries=True,
        allow_new_entries=True,
        overwrite=True
    )

    transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_gen(
        fn_args.train_files,
        transform_graph,
        4)

    eval_dataset = input_gen(
        fn_args.eval_files,
        transform_graph,
        1)
    return tfx.components.TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            'x': train_dataset,
            'validation_data': eval_dataset,
            'steps_per_epoch': fn_args.train_steps,
            'validation_steps': fn_args.eval_steps
        })


def run_fn(fn_args: FnArgs):
    """Train the model based on given args.
    Args:
      fn_args: Holds args used to train the model as name/value pairs.
    """
    # Number of nodes in the first layer of the DNN
    first_dnn_layer_size = 100
    num_dnn_layers = 4
    dnn_decay_factor = 0.7

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_gen(fn_args.train_files,
                              tf_transform_output, 4)
    eval_dataset = input_gen(fn_args.eval_files,
                             tf_transform_output, 1)

    mirrored_strategy = tf.distribute.MirroredStrategy()
    if fn_args.hyperparameters:
        hparams = keras_tuner.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        # _build_keras_model.
        hparams = _get_hyperparameters()
    if not fn_args.base_model:
        with mirrored_strategy.scope():
            model = _build_keras_model(
                # Construct layers sizes with exponetial decay
                hparams=hparams
            )
    else:
        model_dir = f"./pipeline/Trainer/model/227/Format-Serving"
        new_model = tf.keras.models.load_model(model_dir)
        model = _build_keras_model(
            # Construct layers sizes with exponetial decay
            hparams=hparams, base_model=new_model)

    # Write logs to path
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    history = model.fit(
        train_dataset,
        batch_size=32,
        validation_data=eval_dataset,
        epochs=25,
        validation_steps=50,
        callbacks=[tensorboard_callback])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output),

    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
