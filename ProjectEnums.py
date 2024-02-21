from enum import Enum

class PossibleActions(Enum):
    """
    Enumeration of possible actions and their associated output paths.

    This enumeration provides a list of possible actions that can be performed in the project,
    along with their associated output paths. Each action represents a specific task or operation,
    and the output path indicates where the results or outputs of that action should be stored.
    """

    ERA5_Hourly_XGB_RUN = r'ERA5/Hourly/XGB/RUN'
    ERA5_Hourly_XGB_LOGO = r'ERA5/Hourly/XGB/LOGO'
    ERA5_Hourly_NN_RUN = r'ERA5/Hourly/NN/RUN'
    ERA5_Hourly_NN_LOGO = r'ERA5/Hourly/NN/LOGO'
    ERA5_Daily_XGB_RUN = r'ERA5/Daily/XGB/RUN'
    ERA5_Daily_XGB_LOGO = r'ERA5/Daily/XGB/LOGO'
    ERA5_Daily_NN_RUN = r'ERA5/Daily/NN/RUN'
    ERA5_Daily_NN_LOGO = r'ERA5/Daily/NN/LOGO'
    CIMP6_Daily_XGB_RUN = r'CIMP6/Daily/XGB/RUN'
    CIMP6_Daily_XGB_LOGO = r'CIMP6/Daily/XGB/LOGO'
    CIMP6_Daily_NN_RUN = r'CIMP6/Daily/NN/RUN'
    CIMP6_Daily_NN_LOGO = r'CIMP6/Daily/NN/LOGO'


