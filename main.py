import os
from ProjectEnums import PossibleActions
from WeatherDataProcess.CIMP6Processor import CIMP6Processor
from WeatherDataProcess.ERA5Processor import ERA5Processor


def create_output_path(requested_action: PossibleActions) -> str:
    """
    Create an output path based on the requested action.

    Parameters:
        requested_action (PossibleActions): The requested action.

    Returns:
        str: The output path.

    """
    created_path = os.path.join(os.getcwd(), 'Output', requested_action.value)

    if not os.path.exists(created_path):
        os.makedirs(created_path)

    return created_path


def is_ERA5(requested_action: PossibleActions) -> bool:
    """
    Determines whether the requested action corresponds to ERA5 data processing.

    Args:
        requested_action (PossibleActions): The requested action to be performed.

    Returns:
        bool: True if the requested action corresponds to ERA5 data processing,
              False otherwise.
    """
    return requested_action == PossibleActions.ERA5_Hourly_XGB_RUN or \
        requested_action == PossibleActions.ERA5_Hourly_XGB_LOGO or \
        requested_action == PossibleActions.ERA5_Hourly_NN_RUN or \
        requested_action == PossibleActions.ERA5_Hourly_NN_LOGO or \
        requested_action == PossibleActions.ERA5_Daily_XGB_RUN or \
        requested_action == PossibleActions.ERA5_Daily_XGB_LOGO or \
        requested_action == PossibleActions.ERA5_Daily_NN_RUN or \
        requested_action == PossibleActions.ERA5_Daily_NN_LOGO  #


def is_hourly(requested_action: PossibleActions) -> bool:
    """
    Determines whether the requested action corresponds to hourly data processing.

    Args:
        requested_action (PossibleActions): The requested action to be performed.

    Returns:
        bool: True if the requested action corresponds to processing hourly data,
              False otherwise.
    """

    return requested_action == PossibleActions.ERA5_Hourly_XGB_RUN or \
        requested_action == PossibleActions.ERA5_Hourly_XGB_LOGO or \
        requested_action == PossibleActions.ERA5_Hourly_NN_RUN or \
        requested_action == PossibleActions.ERA5_Hourly_NN_LOGO


def run_requested_action(requested_action: PossibleActions):
    """
    Run the requested action.

    Parameters:
        requested_action (PossibleActions): The requested action.
    """

    # Create an output path based on the requested action
    output_path = create_output_path(requested_action)

    # Check if the requested action involves ERA5 data processing
    if is_ERA5(requested_action):

        # Determine if the ERA5 data is hourly or daily
        is_hourly_data = is_hourly(requested_action)

        # Initialize an ERA5Processor instance based on the data resolution
        ERA5_processor = ERA5Processor(is_hourly_data)

        # Print the result of the reference
        ERA5_processor.print_result_of_reference()

        # Execute the appropriate action based on the requested action

        if requested_action == PossibleActions.ERA5_Hourly_XGB_RUN or requested_action == PossibleActions.ERA5_Daily_XGB_RUN:
            # Run XGB model training
            ERA5_processor.run_xgb(output_path)

        elif requested_action == PossibleActions.ERA5_Hourly_XGB_LOGO or requested_action == PossibleActions.ERA5_Daily_XGB_LOGO:
            # Run XGB model training with LOGO cross-validation
            ERA5_processor.run_logo_xgb(is_hourly_data, output_path)

        elif requested_action == PossibleActions.ERA5_Hourly_NN_RUN or requested_action == PossibleActions.ERA5_Daily_NN_RUN:
            # Run neural network model training
            ERA5_processor.run_nn()

        elif requested_action == PossibleActions.ERA5_Hourly_NN_LOGO or requested_action == PossibleActions.ERA5_Daily_NN_LOGO:
            # Run neural network model training with LOGO cross-validation
            ERA5_processor.run_logo_nn(is_hourly_data, output_path)

    # The requested action involves CIMP6 data processing
    else:
        # Initialize an ERA5Processor instance based on the data resolution
        CIMP6_processor = CIMP6Processor(False)

        # Execute the appropriate action based on the requested action

        if requested_action == PossibleActions.CIMP6_Daily_XGB_RUN:
            # Run XGB model training
            CIMP6_processor.run_xgb(output_path)

        elif requested_action == PossibleActions.CIMP6_Daily_XGB_LOGO:
            # Run XGB model training with LOGO cross-validation
            CIMP6_processor.run_logo_xgb(False, output_path)

        elif requested_action == PossibleActions.CIMP6_Daily_NN_RUN:
            # Run neural network model training
            CIMP6_processor.run_nn()

        elif requested_action == PossibleActions.CIMP6_Daily_NN_LOGO:
            # Run neural network model training with LOGO cross-validation
            CIMP6_processor.run_logo_nn(False, output_path)



def main():
    """
    Main function to prompt user input and execute requested action.
    """
    print(
        "Enter 0 for ERA5_Hourly_XGB_RUN \n"
        "Enter 1 for ERA5_Hourly_XGB_LOGO \n"
        "Enter 2 for ERA5_Hourly_NN_RUN \n"
        "Enter 3 for ERA5_Hourly_NN_LOGO \n"
        "Enter 4 for ERA5_Daily_XGB_RUN \n"
        "Enter 5 for ERA5_Daily_XGB_LOGO \n"
        "Enter 6 for ERA5_Daily_NN_RUN \n"
        "Enter 7 for ERA5_Daily_NN_LOGO \n"
        "Enter 8 for CIMP6_Daily_XGB_RUN \n"
        "Enter 9 for CIMP6_Daily_XGB_LOGO \n"
        "Enter 10 for CIMP6_Daily_NN_RUN \n"
        "Enter 11 for CIMP6_Daily_NN_LOGO \n"
    )

    number = int(input())
    if number < 0 or number > 11:
        print('Invalid input')
        return
    requested_action = list(PossibleActions)[number]
    run_requested_action(requested_action)


if __name__ == "__main__":
    main()
