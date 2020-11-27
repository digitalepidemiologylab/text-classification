import logging
import logging.config
import sys
import os

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'logging_config.ini')

logging.config.fileConfig(
    config_path, defaults=None)  # , disable_existing_loggers=False)


# def get_disabled(self):
#     return self._disabled


# def set_disabled(self, disabled):
#     frame = sys._getframe(1)
#     if disabled:
#         print('{}:{} disabled the {} logger'.format(
#             frame.f_code.co_filename, frame.f_lineno, self.name))
#     self._disabled = disabled


# logging.Logger.disabled = property(get_disabled, set_disabled)
