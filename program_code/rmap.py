from main_code import calculate_rm as crm
from main_code import generate_candidate_sourcelist as gcs
import config
import os


def main():
    input_path = config.INPUT_PATH
    output_path = config.OUTPUT_PATH
    chitable_path = config.CHITABLE_PATH
    taylor17_path = config.TAYLOR17_PATH

    print("\nWelcome to Dr. Jo-Anne Brown's Rotation Measure Analysis Program! (RMAP)")

    print(f'\nMAKE SURE this path points to the program_code directory, or all of the paths below are absolute:')
    print(f'Current Directory: {os.getcwd()}')

    print(f'\nInput Path: {input_path} \nOutput Path: {output_path} \nChi Table Path: {chitable_path} \nTaylor 17 Path: {taylor17_path}')

    run_which = ''
    while run_which.lower() != 'q':
        print('\nWhat would you like to do today?')
        print('S -- Generate a candidate polarised sourcelist')
        print('R -- Analyze rotation measures (a candidate polarised sourcelist must already exist)')
        print('Q -- Quit')
        run_which = input(f'\n-> ')

        if run_which.lower() == 's':
            print('\nRunning generate_candidate_sourcelist.py...\n')
            gcs.main(input_path, output_path, taylor17_path)
        elif run_which.lower() == 'r':
            print('\nRunning calculate_rm.py...\n')
            crm.main(input_path, output_path, chitable_path)
        elif run_which.lower() == 'q':
            print('\nQuitting program')
        else:
            print('\nInvalid input')


if __name__ == '__main__':
    main()
