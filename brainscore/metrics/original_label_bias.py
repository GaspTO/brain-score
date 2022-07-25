from brainscore.metrics import Metric, Score
from brainscore.metrics.accuracy import Accuracy
import numpy as np

class OriginalLabelBias(Metric):
    """
    Evaluates the bias towards the 'original' label relatively to the 
    'conflict' label.
    
    It corresponds to the shape bias in https://arxiv.org/abs/1911.09071
    """
    def __call__(self,source,target):
        assert (source["image_id"] == target["image_id"]).all()
        #remove entrees without cue-conflict
        
        correct_conflict = target["conflict_image_category"] == source.values
        correct_original = target["original_image_category"] == source.values
        correct_conflict_or_original = np.logical_or(correct_conflict,correct_original)
        values = correct_original[correct_conflict_or_original]
        center = values.mean()
        error = values.std()

        score = Score([center, error], coords={'aggregation': ['center', 'error']}, dims=('aggregation',))
        score.attrs[Score.RAW_VALUES_KEY] = values
        return score



