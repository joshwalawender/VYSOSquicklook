"""
This is a template pipeline.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext

# MODIFY THIS IMPORT to reflect the name of the module created in the primitives directory
from vysosdrp.primitives import *


class QuickLookPipeline(BasePipeline):
    """
    The template pipeline.
    """

    # modify the event table to use the events actually defined in the primitives
    event_table = {
        "next_file": ("ReadFITS", "file_ingested", "moon_info"),
        "moon_info": ("MoonInfo", "determining_moon_info", "gain_correct"),
        "gain_correct": ("GainCorrect", "correcting_gain", "create_deviation"),
        "create_deviation": ("CreateDeviation", "estimating_uncertainty", "make_source_mask"),
        "make_source_mask": ("MakeSourceMask", "making_source_mask", "subtract_background"),
        "subtract_background": ("SubtractBackground", "subtracting_background", "extract"),
        "extract": ("ExtractStars", "extracting_stars", "record"),
        "record": ("Record", "recording_results_in_mongo", None),
#         "regenerate_plot": ("RegeneratePlot", "generating_plot", None),
#         "transfer_file": (),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)


class GeneratePlotOnly(BasePipeline):
    """
    The template pipeline.
    """

    # modify the event table to use the events actually defined in the primitives
    event_table = {
        "next_file": ("ReadFITS", "file_ingested", "regenerate_plot"),
        "regenerate_plot": ("RegeneratePlot", "generating_plot", None),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
