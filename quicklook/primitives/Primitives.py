from pathlib import Path

from keckdata import fits_reader, VYSOS20

from keckdrpframework.primitives.base_primitive import BasePrimitive
from keckdrpframework.models.arguments import Arguments

# MODIFY the name of this class and make sure that this module/file is imported in the pipelines definition file that
# has been created in the pipeline directory


class ReadFITS(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.logger = context.pipeline_logger

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        if some_pre_condition:
            self.logger.debug("Precondition for ReadFITS is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.logger.debug("Postcondition for ReadFITS is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.logger.info("Running ReadFITS action")
        fitsfile = Path(self.action.args.name).expanduser()
        if fitsfile.exists():
            self.logger.info(f"File: {fitsfile}")
        else:
            self.logger.info(f"Could not find file: {fitsfile}")
            return False

        # Read FITS file
        self.action.args.kd = ccdproc.fits_reader(file, datatype=VYSOS20)

        return self.action.args


class GainCorrect(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.logger = context.pipeline_logger

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        if some_pre_condition:
            self.logger.debug("Precondition for Template is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.logger.debug("Postcondition for Template is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.logger.info("Running GainCorrect action")

        for i,pd in enumerate(self.action.args.kd.pixeldata):
            self.action.args.kd.pixeldata[i] = ccdproc.gain_correct(pd, self.gain.value)

        return self.action.args


class Template(BasePrimitive):
    """
    This is a template for primitives, which is usually an action.

    The methods in the base class can be overloaded:
    - _pre_condition
    - _post_condition
    - _perform
    - apply
    - __call__
    """

    def __init__(self, action, context):
        """
        Constructor
        """
        BasePrimitive.__init__(self, action, context)
        # to use the pipeline logger instead of the framework logger, use this:
        self.logger = context.pipeline_logger

    def _pre_condition(self):
        """Check for conditions necessary to run this process"""
        some_pre_condition = True

        if some_pre_condition:
            self.logger.debug("Precondition for Template is satisfied")
            return True
        else:
            return False

    def _post_condition(self):
        """Check for conditions necessary to verify that the process run correctly"""
        some_post_condition = True

        if some_post_condition:
            self.logger.debug("Postcondition for Template is satisfied")
            return True
        else:
            return False

    def _perform(self):
        """
        Returns an Argument() with the parameters that depends on this operation.
        """
        self.logger.info("Running Template action")

        return self.action.args

