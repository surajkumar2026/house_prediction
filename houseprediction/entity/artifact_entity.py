from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str
    

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str

@dataclass
class RegressionMetricArtifact:
    r2_score: float
    MSE: float
    RMSE: float
    MAE: float

@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metric_artifact: RegressionMetricArtifact
    test_metric_artifact: RegressionMetricArtifact