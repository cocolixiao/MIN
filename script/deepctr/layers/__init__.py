import tensorflow as tf

from .activation import Dice
from .core import DNN, LocalActivationUnit, PredictionLayer
from .normalization import LayerNormalization
from .sequence import (SequencePoolingLayer, WeightedSequenceLayer,AttentionSequencePoolingLayer,
                       PositionEncoding,AttentionSequencePoolingLayer2,Transformer2)

from .utils import NoMask, Hash, Linear, Add, combined_dnn_input, softmax, reduce_sum

custom_objects = {'tf': tf,
                  'DNN': DNN,
                  'PredictionLayer': PredictionLayer,
                  'LocalActivationUnit': LocalActivationUnit,
                  'Dice': Dice,
                  'SequencePoolingLayer': SequencePoolingLayer,
                  'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                  'LayerNormalization': LayerNormalization,
                  'NoMask': NoMask,
                  'Hash': Hash,
                  'Linear': Linear,
                  'WeightedSequenceLayer': WeightedSequenceLayer,
                  'Add': Add,
                  'softmax': softmax,
                  'reduce_sum': reduce_sum,
                  'PositionEncoding':PositionEncoding,
                  'AttentionSequencePoolingLayer2': AttentionSequencePoolingLayer2,
                  'Transformer2': Transformer2
                  }
