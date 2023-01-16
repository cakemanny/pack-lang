
class PackLangError(Exception):
    def __init__(self, msg, /, location=None):
        if location is not None:
            super().__init__(msg, location)
        else:
            super().__init__(msg)
        self.location = location


class SyntaxError(PackLangError):
    pass


class Unmatched(SyntaxError):
    "something ended that hadn't begun"
    def __init__(self, c, remaining, location=None):
        super().__init__(c, location)
        # c is the character that was unmatched
        self.c = c
        # We store the remaining text on the exception in case it's
        # possible to recover. (In fact that's how we read lists.)
        self.remaining = remaining


class Unclosed(SyntaxError):
    "something started but never ended"


class SemanticError(PackLangError):
    pass


class EvalError(PackLangError):
    pass
