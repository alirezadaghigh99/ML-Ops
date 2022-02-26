import datetime
import os
from typing import List
import tfx.v1 as tfx
import absl
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform

from tfx.components.trainer.executor import Executor
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.orchestration import pipeline
from tfx.dsl.components.common import resolver
from tfx.dsl.experimental import latest_artifacts_resolver
from tfx.proto import pusher_pb2, example_gen_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.orchestration.airflow.airflow_dag_runner import AirflowDagRunner
from tfx.orchestration.airflow.airflow_dag_runner import AirflowPipelineConfig
import pandas as pd

user_provided_schema_path = None
def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, serving_model_dir: str,
                     metadata_path: str,
                     beam_pipeline_args: List[str]) -> pipeline.Pipeline:
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=9),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=3),
            example_gen_pb2.SplitConfig.Split(name='test', hash_buckets=1),

        ]))

    # Ingest the data through ExampleGen
    example_gen = CsvExampleGen(input_base=_data_root, output_config=output)
    statistics_gen = StatisticsGen(
        examples=example_gen.outputs['examples'])
    if user_provided_schema_path:
        # Import user-provided schema.
        schema_gen = tfx.components.ImportSchemaGen(
            schema_file=user_provided_schema_path)
        # Performs anomaly detection based on statistics and data schema.
        example_validator = tfx.components.ExampleValidator(
            statistics=statistics_gen.outputs['statistics'],
                schema=schema_gen.outputs['schema'])
      else:
        # Generates schema based on statistics files.
        schema_gen = tfx.components.SchemaGen(
            statistics=statistics_gen.outputs['statistics'],
            infer_feature_shape=True)
    transform = tfx.components.Transform(
      examples=(example_gen.outputs['examples']),
      schema=schema_gen.outputs['schema'],
      module_file=module_file)
    if False: # TODO Enable tuning
        tuner = tfx.components.Tuner(
        module_file=module_file,
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        train_args=tfx.proto.TrainArgs(num_steps=20),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

  # Uses user-provided Python function that trains a model.
    trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      # If Tuner is in the pipeline, Trainer can take Tuner's output
      # best_hyperparameters artifact as input and utilize it in the user module
      # code.
      #
      # If there isn't Tuner in the pipeline, either use Importer to import
      # a previous Tuner's output to feed to Trainer, or directly use the tuned
      # hyperparameters in user module code and set hyperparameters to None
      # here.
      #
      # Example of Importer,
      #   hparams_importer = Importer(
      #     source_uri='path/to/best_hyperparameters.txt',
      #     artifact_type=HyperParameters).with_id('import_hparams')
      #   ...
      #   hyperparameters = hparams_importer.outputs['result'],
      hyperparameters=(tuner.outputs['best_hyperparameters']
                       if False else None),
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

    model_resolver = resolver.Resolver(
      strategy_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(
          type=ModelBlessing)).with_id('latest_blessed_model_resolver')
    eval_config = tfma.EvalConfig(
      model_specs=[
          tfma.ModelSpec(
              signature_name='serving_default', label_key='Sale',
              preprocessing_function_names=['transform_features'])
      ],
      slicing_specs=[tfma.SlicingSpec()],
      metrics_specs=[
          tfma.MetricsSpec(metrics=[
              tfma.MetricConfig(
                  class_name='AUC',
                  threshold=tfma.MetricThreshold(
                      value_threshold=tfma.GenericValueThreshold(
                          lower_bound={'value': 0.7}),

                      change_threshold=tfma.GenericChangeThreshold(
                          direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                          absolute={'value': -1e-10})))
          ])
          ])
    evaluator = Evaluator(
          examples=example_gen.outputs['examples'],
          model=trainer.outputs['model'],
          baseline_model=model_resolver.outputs['model'],
          # Change threshold will be ignored if there is no baseline (first run).
          eval_config=eval_config)
    pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory='./pipeline/served_models')))
    return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen,
          statistics_gen,
          schema_gen,
          example_validator,
          transform,
          trainer,
          model_resolver,
          evaluator,
          pusher,
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
      beam_pipeline_args=beam_pipeline_args)


if __name__ == "__main__":
    _airflow_config = {
    'schedule_interval': None,
    'start_date': datetime.datetime(2022, 2, 1),
    'depends_on_past': False,
        'email': ['alireza.daghigh199941@gmail.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),

}
    DAG = AirflowDagRunner(AirflowPipelineConfig(_airflow_config)).run(
    _create_pipeline(
        pipeline_name=_pipeline_name,
        pipeline_root=_pipeline_root,
        data_root=_data_root,
        module_file=_module_file,
        serving_model_dir=_serving_model_dir,
        metadata_path=_metadata_path,
        beam_pipeline_args=_beam_pipeline_args))
