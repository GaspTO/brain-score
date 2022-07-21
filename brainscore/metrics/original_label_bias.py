from brainscore.metrics import Metric, Score

class OriginalLabelBias(Metric):
    """
    Evaluates the bias towards the 'original' label relatively to the 
    'conflict' label.
    
    It corresponds to the shape bias in https://arxiv.org/abs/1911.09071
    """
    def __call__(self,source,target):
        conflict_labels = target.sel(category="conflict_image_category")
        original_labels = target.sel(category="original_image_category")
        total_conflict = (source == conflict_labels).sum()
        total_original = (source == original_labels).sum()
        original_bias = total_conflict / (total_conflict + total_original)
        return Score(original_bias)
