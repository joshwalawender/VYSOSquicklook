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
        "update_directory": ("UpdateDirectory", "updating_directory", None),
        "set_overwrite": ("SetOverwrite", "setting_overwrite", None),

        "next_file": ("ReadFITS", "file_ingested", "copy_data_locally"),
        "copy_data_locally": ("CopyDataLocally", "copying_data", "moon_info"),
        "moon_info": ("MoonInfo", "determining_moon_info", "gain_correct"),
        "gain_correct": ("GainCorrect", "correcting_gain", "create_deviation"),
        "create_deviation": ("CreateDeviation", "estimating_uncertainty", "make_source_mask"),
        "make_source_mask": ("MakeSourceMask", "making_source_mask", "subtract_background"),
        "subtract_background": ("SubtractBackground", "subtracting_background", "extract"),
        "extract": ("ExtractStars", "extracting_stars", "solve_astrometry"),
        "solve_astrometry": ("SolveAstrometry", "solving", "get_catalog"),
        "get_catalog": ("GetCatalogStars", "getting_catalog", "associate_stars"),
        "associate_stars": ("AssociateCatalogStars", "associating_catalog", "render_jpeg"),
        "render_jpeg": ("RenderJPEG", "rendering_jpeg", "record"),
        "record": ("Record", "recording_results_in_mongo", None),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
