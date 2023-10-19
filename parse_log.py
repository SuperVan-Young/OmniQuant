from experiments.log import lookup_results
import os

def main():
    for exp_group_name in os.listdir('./output'):
        lookup_results(
            output_path = os.path.join('./output', exp_group_name),
            save_path = os.path.join('./results', f"{exp_group_name}.csv")
        )

if __name__ == '__main__':
    main()