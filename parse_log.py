from experiments.log import lookup_results

def main():
    lookup_results(
        output_path='./output/efficient_grouping',
        save_path='./results/efficient_grouping.csv',
        second_level_type='model',
        first_level_type='experiment',
    )

if __name__ == '__main__':
    main()