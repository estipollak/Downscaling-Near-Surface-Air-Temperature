from enum import Enum

class PossibleActions(Enum):
    """
    Enumeration of possible actions and their associated output paths.

    This enumeration provides a list of possible actions that can be performed in the project,
    along with their associated output paths. Each action represents a specific task or operation,
    and the output path indicates where the results or outputs of that action should be stored.
    """

    ERA5_XGB_RUN = r'ERA5\XGB\RUN'
    ERA5_XGB_LOGO = r'ERA5\XGB\LOGO'
    ERA5_NN_RUN = r'ERA5\NN\RUN'
    ERA5_NN_LOGO = r'ERA5\NN\LOGO'
    CIMP6_XGB_RUN = r'CIMP6\XGB\RUN'
    CIMP6_XGB_LOGO = r'CIMP6\XGB\LOGO'
    CIMP6_NN_RUN = r'CIMP6\NN\RUN'
    CIMP6_NN_LOGO = r'CIMP6\NN\LOGO'