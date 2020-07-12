"""
This is a template pipeline.

"""

from keckdrpframework.pipelines.base_pipeline import BasePipeline
from keckdrpframework.models.processing_context import ProcessingContext

# MODIFY THIS IMPORT to reflect the name of the module created in the primitives directory
from quicklook.primitives.Primitives import ReadFITS, GainCorrect


class QuickLookPipeline(BasePipeline):
    """
    The template pipeline.
    """

    # modify the event table to use the events actually defined in the primitives
    event_table = {
        "ingest_file": ("ReadFITS", "file_ingested", "get_header_info"),
        "gain_correct": ("GainCorrect", "correcting_gain", None),
#         "create_deviation": (),
#         "make_source_mask": (),
#         "subtract_background": (),
#         "solve_astrometry": (),
#         "calculate_pointing_error": (),
#         "extract": (),
#         "determine_FWHM": (),
#         "record": (),
    }

    def __init__(self, context: ProcessingContext):
        """
        Constructor
        """
        BasePipeline.__init__(self, context)
