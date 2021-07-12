"""
global exceptions
"""


class ParamException(Exception):
    """
    Invalid parameter exception
    """

    def __init__(self, msg, fields=None):
        self.fields = fields
        self.msg = msg

    def __str__(self):
        return self.msg
