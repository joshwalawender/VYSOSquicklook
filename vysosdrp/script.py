from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
import subprocess
import time
import argparse
import sys
import traceback
import pkg_resources
import logging.config

# the preferred way to import the pipeline is a direct import

from vysosdrp.quicklook import QuickLookPipeline


def _parseArguments(in_args):
    description = "Template pipeline CLI"

    # this is a simple case where we provide a frame and a configuration file
    parser = argparse.ArgumentParser(prog=f"{in_args[0]}", description=description)
    parser.add_argument('-c', dest="config_file", type=str, help="Configuration file")
    parser.add_argument('-frames', nargs='*', type=str, help='input image file (full path, list ok)', default=None)

    # in this case, we are loading an entire directory, and ingesting all the files in that directory
    parser.add_argument('-d', '--directory', dest="dirname", type=str, help="Input directory", nargs='?', default=None)

    parser.add_argument("-O", "--overwrite", dest="overwrite",
        default=False, action="store_true",
        help="Reprocess files if they already exist in database?")

    args = parser.parse_args(in_args[1:])
    return args


def main():

    args = _parseArguments(sys.argv)

    # START HANDLING OF CONFIGURATION FILES ##########
    pkg = 'vysosdrp'

    # load the framework config file from the config directory of this package
    # this part uses the pkg_resources package to find the full path location
    # of framework.cfg
    framework_config_file = "framework.cfg"
    framework_config_fullpath = pkg_resources.resource_filename(pkg, framework_config_file)

    # load the logger config file from the config directory of this package
    # this part uses the pkg_resources package to find the full path location
    # of logger.cfg
    framework_logcfg_file = 'logger.cfg'
    framework_logcfg_fullpath = pkg_resources.resource_filename(pkg, framework_logcfg_file)

    # add PIPELINE specific config files
    # this part uses the pkg_resource package to find the full path location
    # of template.cfg or uses the one defines in the command line with the option -c
    if args.config_file is None:
        pipeline_config_file = 'pipeline.cfg'
        pipeline_config_fullpath = pkg_resources.resource_filename(pkg, pipeline_config_file)
        pipeline_config = ConfigClass(pipeline_config_fullpath, default_section='DEFAULT')
    else:
        pipeline_config = ConfigClass(args.pipeline_config_file, default_section='DEFAULT')

    if args.overwrite is True:
        pipeline_config.set('VYSOS20', 'overwrite', value='True')

    # END HANDLING OF CONFIGURATION FILES ##########

    try:
        framework = Framework(QuickLookPipeline, framework_config_fullpath)
        logging.config.fileConfig(framework_logcfg_fullpath)
        framework.config.instrument = pipeline_config
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    # this part defines a specific logger for the pipeline, so that
    # we can separate the output of the pipeline
    # from the output of the framework
    framework.context.pipeline_logger = getLogger(framework_logcfg_fullpath, name="pipeline")
    framework.logger = getLogger(framework_logcfg_fullpath, name="DRPF")

    framework.logger.info("Framework initialized")

    # frames processing
    if args.frames is not None:
        for frame in args.frames:
            # ingesting and triggering the default ingestion event specified in the configuration file
            framework.ingest_data(None, args.frames, False)

    # ingest an entire directory, trigger "next_file" on each file, optionally continue to monitor if -m is specified
    elif args.dirname is not None:
        framework.logger.info(f'Ingesting files from {args.dirname}')
        framework.ingest_data(args.dirname, None, True)

    framework.start(False, False, False, args.dirname is not None)


if __name__ == "__main__":
    main()