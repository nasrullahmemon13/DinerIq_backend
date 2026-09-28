"""Portable inference artifact exported from fitted native Spark regressors.

This format stores learned coefficients or complete learned trees, not predictions.
Export must pass prediction parity against the native Spark model before publishing.
"""
import numpy as np


def _node(node):
    result = {'prediction': float(node.prediction())}
    if node.getClass().getSimpleName() == 'InternalNode':
        split = node.split()
        if split.getClass().getSimpleName() != 'ContinuousSplit':
            raise ValueError('Only continuous splits supported by portable format')
        result.update(feature=int(split.featureIndex()), threshold=float(split.threshold()),
                      left=_node(node.leftChild()), right=_node(node.rightChild()))
    return result


def export_model(model, features):
    result = {'format': 'dineiq.spark.regression.v1', 'features': features,
              'spark_class': type(model).__name__, 'uid': model.uid}
    if hasattr(model, 'coefficients'):
        result.update(kind='linear', coefficients=model.coefficients.toArray().tolist(), intercept=float(model.intercept))
    elif hasattr(model, 'trees'):
        result.update(kind='forest', trees=[_node(t._java_obj.rootNode()) for t in model.trees])
    else:
        result.update(kind='tree', tree=_node(model._java_obj.rootNode()))
    return result


class PortableSparkModel:
    def __init__(self, artifact):
        if artifact.get('format') != 'dineiq.spark.regression.v1':
            raise ValueError('Unsupported Spark artifact format')
        self.artifact = artifact

    def predict(self, frame):
        values = np.asarray(frame[self.artifact['features']], dtype=float)
        if self.artifact['kind'] == 'linear':
            return values @ np.asarray(self.artifact['coefficients']) + self.artifact['intercept']
        def walk(node, row):
            while 'feature' in node:
                node = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
            return node['prediction']
        trees = self.artifact.get('trees', [self.artifact.get('tree')])
        return np.asarray([sum(walk(t, row) for t in trees)/len(trees) for row in values])
