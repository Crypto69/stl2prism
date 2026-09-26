"""One readable sentence for an exception, for the messages the UI shows.

A user who dropped a file on the app should read "the file is not a valid
STL" and not "ValueError: could not convert string to float: 'nan'". The
raw text is kept where it says something (a pipeline RuntimeError names
what failed); the common low-level failures get a plain explanation.
"""
import errno


def describe(exc):
    """A short, user-readable description of `exc`."""
    # an exception written as a sentence for the user (Blueprint's) is kept as is
    if getattr(exc, 'readable', False) and str(exc).strip():
        return str(exc).strip()
    if isinstance(exc, MemoryError):
        return ('the computer ran out of memory. Convert fewer bodies at once, '
                'close other programs, or use a coarser tolerance')
    if isinstance(exc, RecursionError):
        return 'the mesh is too tangled for the fitter (recursion limit reached)'
    if isinstance(exc, TimeoutError):
        return str(exc) or 'it took too long and was stopped'
    if isinstance(exc, PermissionError):
        return f'permission denied writing {getattr(exc, "filename", None) or "a file"}'
    if isinstance(exc, FileNotFoundError):
        return f'a file went missing ({getattr(exc, "filename", None) or "unknown"})'
    if isinstance(exc, OSError):
        if exc.errno == errno.ENOSPC:
            return 'the disk is full'
        if exc.errno == errno.EACCES:
            return 'permission denied'
        return f'a system error ({exc.strerror or exc})' if exc.strerror else str(exc)
    if isinstance(exc, (UnicodeDecodeError, UnicodeError)):
        return 'the file is not text in the expected encoding'
    if isinstance(exc, KeyboardInterrupt):
        return 'it was interrupted'
    name = type(exc).__name__
    text = str(exc).strip()
    # library-level errors whose class name means nothing to a user keep
    # only their text; a bare class name alone is worse than nothing
    if name in ('ValueError', 'RuntimeError', 'PrepError', 'AssertionError', 'KeyError',
                'IndexError', 'TypeError', 'AttributeError', 'ZeroDivisionError',
                'NotImplementedError', 'StopIteration', 'Exception'):
        if not text:
            return f'an internal error ({name}) with no message; the log has the traceback'
        if name in ('KeyError', 'IndexError', 'TypeError', 'AttributeError',
                    'ZeroDivisionError', 'AssertionError'):
            return f'an internal error ({name}: {text}); the log has the traceback'
        return text
    return f'{name}: {text}' if text else name
