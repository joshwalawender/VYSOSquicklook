"""
This is a template pipeline.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext

# MODIFY THIS IMPORT to reflect the name of the module created in the primitives directory
from quicklook.primitives.Primitives import *


class QuickLookPipeline(BasePipeline):
    """
    The template pipeline.
    """

    # modify the event table to use the events actually defined in the primitives
    event_table = {
        "next_file": ("ReadFITS", "file_ingested", "gain_correct"),
        "gain_correct": ("GainCorrect", "correcting_gain", "create_deviation"),
        "create_deviation": ("CreateDeviation", "estimating_uncertainty", "make_source_mask"),
        "make_source_mask": ("MakeSourceMask", "making_source_mask", "subtract_background"),
        "subtract_background": ("SubtractBackground", "subtracting_background", "extract"),
        "extract": ("ExtractStars", "extracting_stars", "record"),
#         "solve_astrometry": (),
#         "calculate_pointing_error": (),
        "record": ("Record", "recording_results_in_mongo", None),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
