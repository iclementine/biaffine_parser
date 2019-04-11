import os
import logging
import argparse
import configparser

def parser_arg_cfg():
    # Declare expected command-line arguments
    argp = argparse.ArgumentParser()

    # Add positional arguments (many)
    # argp.add_argument ("inputs", metavar='INFILE', type=str, nargs='+',
    #     help='Provide input logs to parse and load into database');

    # Add an optional string argument 'config' 
    argp.add_argument ("-c", "--config", dest='config_file', default='default.cfg', type=str)
    # Add a optional switch (boolean optional argument)
    argp.add_argument ("-t", "--test", dest='test', default=False, action='store_true',
        help='Only test')
    argp.add_argument ("-v", "--verbose", dest='verbose', default=False, action='store_true',
        help='Be verbose')

    # Parse command line    
    args, _ = argp.parse_known_args()

    # Parse a config ini-style file
    config_file = args.config_file
    logging.info('Reading configuration from %s' %(config_file))
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_file)

    # Do something with them, type-cast should be done by your own
    if args.verbose:
        logging.info('Will produce verbose output')
    return args, config


