"""
This is a template pipeline.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext

# MODIFY THIS IMPORT to reflect the name of the module created in the primitives directory
from vysosdrp.primitives import *


class QuickLookPipeline(BasePipeline):
    """
    Quick look at VYSOS data.
    """

    # modify the event table to use the events actually defined in the primitives
    event_table = {
        # One time use utilities
        "update_directory": ("UpdateDirectory", "updating_directory", None),
        "set_overwrite": ("SetOverwrite", "setting_overwrite", None),

        # Ingest a file
        "next_file": ("ReadFITS", "reading_file", "sorting_hat"),
        "sorting_hat": ("sort_file", "sorting", None),

        # Science data quick look
        "prepare_science_file": ("PrepareScience", "preparing_science", "copy_data_locally"),
        "copy_data_locally": ("CopyDataLocally", "copying_data", "moon_info"),
        "moon_info": ("MoonInfo", "determining_moon_info", "gain_correct"),
        "gain_correct": ("GainCorrect", "correcting_gain", "create_deviation"),
        "create_deviation": ("CreateDeviation", "estimating_uncertainty", "make_source_mask"),
        "make_source_mask": ("MakeSourceMask", "making_source_mask", "create_background"),
        "create_background": ("CreateBackground", "creating_background", "extract"),
        "extract": ("ExtractStars", "extracting_stars", "solve_astrometry"),
        "solve_astrometry": ("SolveAstrometry", "solving", "get_catalog"),
        "get_catalog": ("GetCatalogStars", "getting_catalog", "associate_stars"),
        "associate_stars": ("AssociateCatalogStars", "associating_catalog", "render_jpeg"),
        "render_jpeg": ("RenderJPEG", "rendering_jpeg", "record"),
        "record": ("Record", "recording_results_in_mongo", None),

        # Bias processing
        "copy_bias_locally": ("CopyDataLocally", "copying_data", "save_bias_to_list"),
        "save_bias_to_list": ("SaveToList", "saving_result", "record_bias"),
        "record_bias": ("Record", "recording_results_in_mongo", "stack_bias_frames"),
        "stack_bias_frames": ("StackBiases", "stacking_biases", None),

        # Dark processing
        "copy_dark_locally": ("CopyDataLocally", "copying_data", "save_dark_to_list"),
        "save_dark_to_list": ("SaveToList", "saving_result", "record_dark"),
        "record_dark": ("Record", "recording_results_in_mongo", None),

        # Flat processing
        "copy_flat_locally": ("CopyDataLocally", "copying_data", "flat_gain_correct"),
        "flat_gain_correct": ("GainCorrect", "correcting_gain", "flat_create_deviation"),
        "flat_create_deviation": ("CreateDeviation", "estimating_uncertainty", "flat_stats"),
        "flat_stats": ("ImageStats", "flat_stats", "render_jpeg"),
        "render_jpeg": ("RenderJPEG", "rendering_jpeg", "record_flat"),
        "record_flat": ("Record", "recording_results_in_mongo", "save_flat_to_list"),
        "save_flat_to_list": ("SaveToList", "saving_result", None),

    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        context.biases = list()
        context.darks = list()
        context.flats = dict()

    def sort_file(self, action, context):
        """
        Decide which branch of pipeline to go to.
        """
        # Determine image type
        imtype = action.args.kd.type()
        context.pipeline_logger.info(f"File has IMTYPE '{imtype}'")
        if imtype == 'OBJECT':
            context.push_event("prepare_science_file", action.args)
        elif imtype == 'BIAS':
            context.push_event("copy_bias_locally", action.args)
        elif imtype == 'DARK':
            context.push_event("copy_dark_locally", action.args)
        elif imtype == 'FLAT':
            context.push_event("copy_flat_locally", action.args)
        else:
            self.log.error(f"Unable to interpret image type: {imtype}")
            raise TypeError
