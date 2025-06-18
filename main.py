import argparse
import sys

def run_augmentation_selection():
    from cell_segmentation.forward_selection_integration import run_augmentation_selection_experiment
    run_augmentation_selection_experiment()

def run_augmentation_amount():
    from cell_segmentation.augmentation_amount_experiment import main as aug_amount_main
    aug_amount_main()

def run_loss_function():
    from cell_segmentation.loss_function_cross_validation import main as loss_fn_main
    loss_fn_main()

def run_model_configuration():
    from cell_segmentation.model_configuration_experiment import main as model_config_main
    model_config_main()

def main():
    parser = argparse.ArgumentParser(description="Cell Segmentation Experiment Launcher")
    parser.add_argument('--experiment', type=str, required=True,
                        choices=['augmentation_selection', 'augmentation_amount', 'loss_function', 'model_configuration'],
                        help='Which experiment to run')
    args = parser.parse_args()

    if args.experiment == 'augmentation_selection':
        run_augmentation_selection()
    elif args.experiment == 'augmentation_amount':
        run_augmentation_amount()
    elif args.experiment == 'loss_function':
        run_loss_function()
    elif args.experiment == 'model_configuration':
        run_model_configuration()
    else:
        print("Unknown experiment. Use --help for options.")
        sys.exit(1)

if __name__ == "__main__":
    main()