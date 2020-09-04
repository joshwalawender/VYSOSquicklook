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
        "prepare_science_file": ("PrepareScience", "preparing_science", "moon_info"),
#         "copy_data_locally": ("CopyDataLocally", "copying_data", "moon_info"),
        "moon_info": ("MoonInfo", "determining_moon_info", "subtract_bias"),
        "subtract_bias": ("BiasSubtract", "subtracting_bias", "subtract_dark"),
        "subtract_dark": ("DarkSubtract", "subtracting_dark", "gain_correct"),
        "gain_correct": ("GainCorrect", "correcting_gain", "create_deviation"),
        "create_deviation": ("CreateDeviation", "estimating_uncertainty", "make_source_mask"),
        "make_source_mask": ("MakeSourceMask", "making_source_mask", "create_background"),
        "create_background": ("CreateBackground", "creating_background", "extract"),
        "extract": ("ExtractStars", "extracting_stars", "solve_astrometry"),
        "solve_astrometry": ("SolveAstrometry", "solving", "get_calibrators"),
        "get_calibrators": ("GetCalibrationStars", "getting_calibrators", "associate_calibrators"),
        "associate_calibrators": ("AssociateCalibratorStars", "associating_calibrators", "render_jpeg"),

        "get_targets": ("GetTargetStars", "getting_targets", "associate_targets"),
        "associate_targets": ("AssociateTargetStars", "associating_targets", "render_jpeg"),

        "render_jpeg": ("RenderJPEG", "rendering_jpeg", "record"),
        "record": ("Record", "recording_results_in_mongo", None),

        # Bias processing
#         "copy_bias_locally": ("CopyDataLocally", "copying_data", "save_bias_to_list"),
        "save_bias_to_list": ("SaveToList", "saving_result", "bias_stats"),
        "bias_stats": ("ImageStats", "bias_stats", "bias_render_jpeg"),
        "bias_render_jpeg": ("RenderJPEG", "rendering_jpeg", "record_bias"),
        "record_bias": ("Record", "recording_results_in_mongo", "stack_bias_frames"),
        "stack_bias_frames": ("StackBiases", "stacking_biases", None),

        # Dark processing
#         "copy_dark_locally": ("CopyDataLocally", "copying_data", "save_dark_to_list"),
        "save_dark_to_list": ("SaveToList", "saving_result", "bias_correct_dark_frames"),
        "bias_correct_dark_frames": ("BiasSubtract", "bias_correcting_darks", "dark_stats"),
        "dark_stats": ("ImageStats", "dark_stats", "dark_render_jpeg"),
        "dark_render_jpeg": ("RenderJPEG", "rendering_jpeg", "record_dark"),
        "record_dark": ("Record", "recording_results_in_mongo", "stack_dark_frames"),
        "stack_dark_frames": ("StackDarks", "saving_result", None),

        # Flat processing
#         "copy_flat_locally": ("CopyDataLocally", "copying_data", "flat_gain_correct"),
        "flat_gain_correct": ("GainCorrect", "correcting_gain", "flat_create_deviation"),
        "flat_create_deviation": ("CreateDeviation", "estimating_uncertainty", "flat_stats"),
        "flat_stats": ("ImageStats", "flat_stats", "flat_render_jpeg"),
        "flat_render_jpeg": ("RenderJPEG", "rendering_jpeg", "save_flat_to_list"),
        "save_flat_to_list": ("SaveToList", "saving_result", "record_flat"),
        "record_flat": ("Record", "recording_results_in_mongo", None),

    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
        context.biases = list()
        context.darks = dict()
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
            context.push_event("save_bias_to_list", action.args)
        elif imtype == 'DARK':
            context.push_event("save_dark_to_list", action.args)
        elif imtype == 'FLAT':
            context.push_event("flat_gain_correct", action.args)
        else:
            self.log.error(f"Unable to interpret image type: {imtype}")
            raise TypeError
