from visdom import Visdom
import socket
import logging

class Viz(object):
    def __init__(self, env_name, port=8097, disabled=False):
        self.disabled = disabled
        self.logger = logging.getLogger(__name__)
        self.opts = {}
        if self.disabled:
            return
        if self.socket_is_used(port=port):
            self.viz = Visdom(port=port, env=env_name)
            if not self.viz.check_connection():
                self.disable()
        else:
            self.disable()

    def disable(self):
        self.logger.info('Could not connect to Visdom server. Make sure Visdom is running in separate tab if you want to use Visdom.')
        self.disabled = True

    def socket_is_used(self, port=8097, hostname='localhost'):
        """Small hack to check whether vizdom server is running"""
        is_used = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((hostname, port))
        except socket.error:
            is_used = True
        finally:
            s.close()
        return is_used

    def add_viz(self, name, xlabel, ylabel):
        if self.disabled:
            return
        if self.viz.win_exists(name):
            self.viz.close(name)
        self.opts[name] = dict(
                xlabel=xlabel,
                ylabel=ylabel,
                title=name
                )

    def update_line(self, name, x, y):
        if self.disabled:
            return
        self.viz.line(Y=y, X=x, win=name, update='append', opts=self.opts[name])
