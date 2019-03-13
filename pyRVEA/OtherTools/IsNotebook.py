
def IsNotebook() -> bool:
    """Checks if the current environment is a Jupyter Notebook or a console.

    Returns
    -------
    bool
        True if notebook. False if console
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
