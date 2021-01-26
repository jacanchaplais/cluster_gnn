from os.path import dirname, basename, isfile, join
import glob


modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = []

for f in modules:
    if (isfile(f) and not (f.endswith('__init__.py') or f.endswith('.py.swp'))):
        __all__.append(basename(f)[:-3])
