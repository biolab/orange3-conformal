import sphinx


def skip_member(app, what, name, obj, skip, options):
    # skip special members except for __init__ and __call__
    return sphinx.ext.napoleon._skip_member(app, what, name, obj, skip, options) or \
           name[:2] == '__' and name not in ['__init__', '__call__', '__next__', '__iter__']


def setup(app):
    # wrap napoleon's skip-member
    app.connect('autodoc-skip-member', skip_member)
    return {'version': sphinx.__display_version__}
