def unindent(s: str) -> str:
    """
    Removing indentation from triple quoted strings.
    """
    lines = s.splitlines()
    if lines[0] == '':
        del lines[0]
    prefix = min(len(l) - len(l.lstrip()) for l in lines)
    return '\n'.join(l[prefix:] for l in lines)
