from logging import getLogger, config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters':
        {
            'default':
                {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
        },
    'handlers':
        {
            'stdout':
                {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                    'stream': 'ext://sys.stdout'
                }
        },
    'loggers':
        {
            '':
                {
                    'handlers': ['stdout'],
                    'level': 'INFO',
                    'propagate': True
                }
        }
}


def get_logger(name):
    config.dictConfig(LOGGING)
    return getLogger(name)
