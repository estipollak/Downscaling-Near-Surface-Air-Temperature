import os
from ProjectEnums import PossibleActions
from WeatherDataProcess.ERA5HourlyProcessor import ERA5HourlyProcessor


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


def run_requested_action(requested_action: PossibleActions):
    """
    Run the requested action.

    Parameters:
        requested_action (PossibleActions): The requested action.

    """
    a = ERA5HourlyProcessor()
    output_path = create_output_path(requested_action)
    a.print_result_of_reference()

    if requested_action == PossibleActions.ERA5_XGB_RUN:
        a.run_xgb(output_path)

    elif requested_action == PossibleActions.ERA5_XGB_LOGO:
        a.run_logo_xgb(output_path)

    elif requested_action == PossibleActions.ERA5_NN_RUN:
        a.run_nn()

    elif requested_action == PossibleActions.ERA5_NN_LOGO:
        a.run_logo_nn(output_path)


def main():
    """
       Main function to prompt user input and execute requested action.
    """
    print(
        "Enter 0 for ERA5_XGB_RUN \n Enter 1 for ERA5_XGB_LOGO \n Enter 2 for ERA5_NN_RUN \n Enter 3 for ERA5_NN_LOGO \n ")

    number = int(input())
    if number < 0 or number > 3:
        print('Invalid input')
        return
    requested_action = list(PossibleActions)[number]
    run_requested_action(requested_action)


if __name__ == "__main__":
    main()
