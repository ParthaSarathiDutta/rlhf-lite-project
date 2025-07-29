import logging

def setup_logging(logfile='run.log'):
    logging.basicConfig(
        filename=logfile,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    return logging.getLogger(__name__)
