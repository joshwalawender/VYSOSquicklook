from keckdrpframework.core.framework import Framework
from keckdrpframework.config.framework_config import ConfigClass
from keckdrpframework.models.arguments import Arguments
from keckdrpframework.utils.drpf_logger import getLogger
from keckdrpframework.tools.interface import FrameworkInterface, Arguments, Event, Framework, ConfigClass
from keckdrpframework.core import queues
import subprocess
import time
import argparse
import sys
import traceback
import pkg_resources
import logging.config
from pathlib import Path
from datetime import datetime
from glob import glob

from vysosdrp.pipelines import QuickLookPipeline


def _parseArguments(in_args):
    parser = argparse.ArgumentParser(prog=f"{in_args[0]}",
                      description='')
    parser.add_argument('-c', dest="config_file", type=str,
           help="Configuration file")
    parser.add_argument("-O", "--overwrite", dest="overwrite",
           default=False, action="store_true",
           help="Reprocess files if they already exist in database?")
    parser.add_argument("--norecord", dest="norecord",
           default=False, action="store_true",
           help="Skip recording results to mongo DB?")
    parser.add_argument('input', type=str, nargs='?',
           help="input image file (full path)", default='')
    args = parser.parse_args(in_args[1:])

    return args


def setup_framework(args, pipeline=QuickLookPipeline):
    # START HANDLING OF CONFIGURATION FILES ##########
    pkg = 'vysosdrp'
    framework_config_file = "framework.cfg"
    framework_config_fullpath = pkg_resources.resource_filename(pkg, framework_config_file)

    framework_logcfg_file = 'logger.cfg'
    framework_logcfg_fullpath = pkg_resources.resource_filename(pkg, framework_logcfg_file)

    # add PIPELINE specific config files
    if args.config_file is None:
        pipeline_config_file = 'pipeline.cfg'
        pipeline_config_fullpath = pkg_resources.resource_filename(pkg, pipeline_config_file)
        pipeline_config = ConfigClass(pipeline_config_fullpath, default_section='DEFAULT')
    else:
        pipeline_config = ConfigClass(args.pipeline_config_file, default_section='DEFAULT')

    if args.overwrite is True:
        pipeline_config.set('VYSOS20', 'overwrite', value='True')
    if args.norecord is True:
        pipeline_config.set('VYSOS20', 'norecord', value='True')

    # END HANDLING OF CONFIGURATION FILES ##########

    try:
        framework = Framework(QuickLookPipeline, framework_config_fullpath)
        logging.config.fileConfig(framework_logcfg_fullpath)
        framework.config.instrument = pipeline_config
    except Exception as e:
        print("Failed to initialize framework, exiting ...", e)
        traceback.print_exc()
        sys.exit(1)

    # this part defines a specific logger for the pipeline, so that we can
    # separate the output of the pipeline from the output of the framework
    framework.context.pipeline_logger = getLogger(framework_logcfg_fullpath, name="pipeline")
    framework.logger = getLogger(framework_logcfg_fullpath, name="DRPF")
    framework.logger.info("Framework initialized")

    return framework


def analyze_one():
    args = _parseArguments(sys.argv)
    p = Path(args.input).expanduser().absolute()
    if p.exists() is False:
        print(f'Unable to find file: {p}')
        return
    args.name = f"{p}"

    framework_config_fullpath = pkg_resources.resource_filename("vysosdrp", "framework.cfg")
    cfg = ConfigClass(framework_config_fullpath)
    queue = queues.get_event_queue(cfg.queue_manager_hostname,
                                   cfg.queue_manager_portnr,
                                   cfg.queue_manager_auth_code)
    if queue is None:
        print("Failed to connect to Queue Manager")
    else:
        pending = queue.get_pending()
        event = Event("next_file", args)
        queue.put(event)


def watch_directory():
    args = _parseArguments(sys.argv)
    framework = setup_framework(args)

    if args.input is not '':
        p = Path(args.input).expanduser()
        args.input = str(p)
        if p.exists() is False:
            framework.context.pipeline_logger.error(f'Could not find: {args.input}')
        else:
            base_path = [x for x in p.parents][-3]
            if base_path == Path('/Users/vysosuser'):
                pass
            elif base_path == Path('/Volumes/VYSOSData'):
                framework.context.pipeline_logger.info(f'Setting file_type to *.fz')
                framework.config['DEFAULT']['file_type'] = '*.fz'

            if p.is_file() is True:
                framework.context.pipeline_logger.info(f'Found file: {args.input}')
            elif p.is_dir() is True:
                framework.context.pipeline_logger.info(f'Found directory: {args.input}')
    else:
#         date_string = datetime.utcnow().strftime('%Y%m%dUT')
#         p = Path(f'~/V20Data/Images/{date_string}').expanduser()
        p = Path(f'~/V20Data/Images').expanduser()
        if p.exists() is False:
            p.mkdir(parents=True, exist_ok=True)
        args.input = str(p)

    framework.logger.info(f'Ingesting files from {args.input}')
    infiles = glob(f"{args.input}/{framework.config['DEFAULT']['file_type']}")
    framework.ingest_data(args.input, infiles, True)
    framework.start(False, False, False, True)


def change_directory():
    args = _parseArguments(sys.argv)
    if args.input is not '':
        newdir = Path(args.input).expanduser()
    else:
        date_string = datetime.utcnow().strftime('%Y%m%dUT')
        newdir = Path(f'~/V20Data/Images/{date_string}').expanduser()
    args.input = str(newdir)
    if newdir.exists() is False:
        newdir.mkdir(parents=True)

    framework_config_fullpath = pkg_resources.resource_filename("vysosdrp", "framework.cfg")
    cfg = ConfigClass(framework_config_fullpath)
    queue = queues.get_event_queue(cfg.queue_manager_hostname,
                                   cfg.queue_manager_portnr,
                                   cfg.queue_manager_auth_code)
    if queue is None:
        print("Failed to connect to Queue Manager")
    else:
        pending = queue.get_pending()
        event = Event("update_directory", args)
        queue.put(event)


if __name__ == "__main__":
    analyze_one()
