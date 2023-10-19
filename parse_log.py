from experiments.log import lookup_results

def main():
    lookup_results(
        output_path='./output/efficient_grouping',
        save_path='./results/efficient_grouping.csv',
    )

    lookup_results(
        output_path='./output/efficient_grouping_ol1p64',
        save_path='./results/efficient_grouping_ol1p64.csv',
    )

if __name__ == '__main__':
    main()