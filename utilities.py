import fnmatch
import inspect
import os

import ruamel.yaml
from ruamel.yaml.comments import CommentedMap, CommentedSeq

# setup yaml parser to preserve quotes, disable aliasing (cross-referencing) of
# duplicate data and write None as ~ (not 'null'); also preserves comments by
# default.
ym = ruamel.yaml.YAML()
ym.preserve_quotes = True
ym.representer.ignore_aliases = lambda x: True
ym.representer.add_representer(
    type(None), lambda self, data: self.represent_scalar("tag:yaml.org,2002:null", "~")
)


def read_yaml(file, quiet=False):
    with open(file, "r") as f:
        yaml_obj = ym.load(f)
    if not quiet:
        print(f"read {file}.")
    return yaml_obj


def write_yaml(yaml_obj, file, quiet=False):
    with open(file, "w") as f:
        ym.dump(yaml_obj, f)
    if not quiet:
        print(f"updated {file}.")


def delete_yaml_keys(yaml_root, path, quiet=False):
    """
    Delete keys in a ruamel.yaml mapping following a path that may include '*'
    (wildcard for 'all keys' at that level). For each matched endpoint:
      - remove the value AND any comment attached to that key
      - print a message like "deleted a:b:c" and append ": *" if the deleted
        value was a container (mapping/sequence).

    Parameters
    ----------
    yaml_root : CommentedMap | dict
        The ruamel.yaml (comment-preserving) root mapping.
    path : list
        A list of path segments, e.g. ['settings_management', '*', 'all_cases', 'MinCapReq'].
        Use '*' to mean "all keys" at that level.
    quiet : bool
        If False, print a deletion line for each removed key.
    """

    def _recurse(obj, path, trail, parent=None):
        # object to recurse into, path to look for, trail we followed to get here
        if not path:
            # end of the line, delete last element of trail from obj
            key = trail[-1]
            try:
                # remove comment attached to this key (if any)
                parent.ca.items.pop(key, None)
            except AttributeError:  # rare
                pass
            # remove the entry itself
            val = parent.pop(key)
            if not quiet:
                msg = f"deleted {': '.join(str(p) for p in trail)}"
                if isinstance(val, (CommentedMap, CommentedSeq, dict, list)):
                    msg += ": *"
                print(msg)
            return

        head, *tail = path

        if isinstance(obj, (CommentedMap, dict)):
            # iterate over a snapshot of matching keys (we may delete under children)
            for k in list(obj.keys()):
                if fnmatch.fnmatch(str(k), str(head)):
                    # match on name or wildcard (*, ?)
                    _recurse(obj[k], tail, trail + [k], obj)

        elif isinstance(obj, (CommentedSeq, list, tuple)):
            # iterate over the specified slice, either "*" for all indices
            # or a single int or a slice object (probably going overboard here)
            if isinstance(head, int):
                s = slice(head, None)
            elif head == "*":
                s = slice()
            elif isinstance(head, slice):
                s = head
            else:
                # could add code here to handle special cases like "3:5"
                raise ValueError(f"Unexpected index for sequence: {head}")

            # process selected elements of obj
            for idx in range(*s.indices(len(obj))):
                _recurse(obj[idx], tail, trail + [idx], obj)

    _recurse(yaml_root, list(path), [])


def add_yaml_key(yaml_root, path, value, quiet=False):
    """
    Add the specified value to a ruamel.yaml mapping at the specified path,
    creating nodes needed to along the way. Also add a comment to the key,
    "managed by <caller_filename>.py"

    Parameters
    ----------
    yaml_root : CommentedMap | dict
        The ruamel.yaml (comment-preserving) root mapping.
    path : list
        A list of path segments, e.g. ['settings_management', 2025, 'all_cases', 'MinCapReq'].
    value : object
       The value to be stored at that location, possibly a dict or sequence.
    quiet : bool
        If False, print a message for each added key.
    """
    obj = yaml_root
    # find the parent node, creating any needed along the way
    for node in path[:-1]:
        obj = obj.setdefault(node, CommentedMap())

    obj[path[-1]] = value
    obj.yaml_add_eol_comment(f"managed by {get_caller_filename()}", path[-1])
    if not quiet:
        msg = f"added {': '.join(str(p) for p in path)}"
        if isinstance(value, (CommentedMap, CommentedSeq, dict, list)):
            msg += ": *"
        print(msg)


def get_caller_filename():
    # Get the caller's stack frame (index 2 = caller of caller of this function)
    try:
        frame = inspect.stack()[2]
        filename = frame.filename  # full path to calling file
        return os.path.basename(filename)  # strip path → just the file name
    except:
        return "unknown script"
