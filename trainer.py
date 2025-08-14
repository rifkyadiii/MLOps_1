# trainer.py

import json
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
from tfx.components.tuner.component import TunerFnResult

LABEL_KEY = 'price_range'
BATCH_SIZE = 64

# UTIL DATASET (BACA TFRECORD GZIP + PARSE)

def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")

def _input_fn(file_pattern, tf_transform_output, batch_size=BATCH_SIZE, shuffle=False):
    """
    Membaca TFRecord (GZIP) hasil Transform, parse pakai transformed_feature_spec,
    lalu memisahkan label.
    """
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    # Tambahkan spec untuk label RAW agar bisa dipop saat parsing
    feature_spec[LABEL_KEY] = tf.io.FixedLenFeature([], tf.int64)

    def _parse_batch(serialized_batch):
        example = tf.io.parse_example(serialized_batch, feature_spec)
        label = example.pop(LABEL_KEY)
        return example, label

    files = tf.io.gfile.glob(file_pattern)
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.interleave(lambda f: _gzip_reader_fn(f),
                       cycle_length=tf.data.AUTOTUNE,
                       num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    ds = ds.batch(batch_size)
    ds = ds.map(_parse_batch, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    # Biarkan TFX mengontrol steps lewat fn_args.(train|eval)_steps
    return ds.repeat()

# SERVING SIGNATURE

def _get_serve_tf_examples_fn(model, tf_transform_output):
    """
    Serving: terima serialized tf.Example RAW -> apply Transform -> prediksi.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
    def serve_tf_examples_fn(serialized_tf_examples):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(LABEL_KEY, None)  # label tidak dipakai saat serving
        parsed = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        transformed = model.tft_layer(parsed)
        return model(transformed)

    return serve_tf_examples_fn

# MODEL BUILDER

def _infer_feature_keys(tf_transform_output):
    """
    Ambil nama fitur TERTRANSFORMASI dari spec. (label tidak termasuk)
    """
    keys = list(tft.TFTransformOutput(tf_transform_output.transform_savedmodel_dir).transformed_feature_spec().keys()) \
        if isinstance(tf_transform_output, str) else list(tf_transform_output.transformed_feature_spec().keys())
    if LABEL_KEY in keys:  # biasanya tidak ada
        keys.remove(LABEL_KEY)
    return keys

def model_builder(hp: HyperParameters, transform_graph_path: str) -> tf.keras.Model:
    """
    Bangun model; input names = fitur TERTRANSFORMASI.
    """
    tf_transform_output = tft.TFTransformOutput(transform_graph_path)
    feature_keys = [k for k in tf_transform_output.transformed_feature_spec().keys() if k != LABEL_KEY]

    inputs = {k: tf.keras.layers.Input(shape=(1,), name=k) for k in feature_keys}
    x = tf.keras.layers.Concatenate()(list(inputs.values()))

    units = hp.Int('units', min_value=128, max_value=512, step=32)
    x = tf.keras.layers.Dense(units=units, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(units=int(units // 2), activation='relu')(x)

    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )
    return model

# TFX: TUNER ENTRYPOINT

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """
    Dipanggil oleh komponen TFX Tuner.
    WAJIB mengembalikan TunerFnResult(tuner, fit_kwargs)
    """
    # Dataset untuk tuning
    tft_out = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_ds = _input_fn(fn_args.train_files, tft_out, batch_size=BATCH_SIZE, shuffle=True)
    eval_ds  = _input_fn(fn_args.eval_files,  tft_out, batch_size=BATCH_SIZE, shuffle=False)

    # Hypermodel harus menerima hp
    def build_model(hp):
        return model_builder(hp, fn_args.transform_graph_path)

    tuner = RandomSearch(
        hypermodel=build_model,
        objective='val_sparse_categorical_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory=fn_args.working_dir,          
        project_name='model_tuning',
        overwrite=True,
    )

    fit_kwargs = {
        'x': train_ds,
        'validation_data': eval_ds,
        'steps_per_epoch': fn_args.train_steps,
        'validation_steps': fn_args.eval_steps,
        'epochs': 10,
    }

    return TunerFnResult(tuner=tuner, fit_kwargs=fit_kwargs)

# TFX: TRAINER ENTRYPOINT

def run_fn(fn_args: FnArgs):
    """
    Dipanggil oleh komponen TFX Trainer.
    Mengambil best HP dari Tuner, melatih final model, dan menyimpan model dengan signature.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Dataset final training
    train_ds = _input_fn(fn_args.train_files, tf_transform_output, batch_size=BATCH_SIZE, shuffle=True)
    eval_ds  = _input_fn(fn_args.eval_files,  tf_transform_output, batch_size=BATCH_SIZE, shuffle=False)

    # Ambil best hyperparameters dari Tuner
    # fn_args.hyperparameters bisa berupa dict (TFX baru) atau JSON string (beberapa versi)
    if isinstance(fn_args.hyperparameters, dict):
        best_hp = HyperParameters.from_config(fn_args.hyperparameters)
    else:
        best_hp = HyperParameters.from_config(json.loads(fn_args.hyperparameters))

    model = model_builder(best_hp, fn_args.transform_graph_path)

    model.fit(
        train_ds,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_ds,
        validation_steps=fn_args.eval_steps,
        epochs=25,
    )

    model.save(
        fn_args.serving_model_dir,
        save_format='tf',
        signatures={'serving_default': _get_serve_tf_examples_fn(model, tf_transform_output)},
    )
