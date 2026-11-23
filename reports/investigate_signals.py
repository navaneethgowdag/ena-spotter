import pandas as pd
import numpy as np

def run():
    train = pd.read_csv('data/train_test.csv')
    val = pd.read_csv('data/validation.csv')
    dec = pd.read_csv('data/december_chart_inputs.csv')

    print("=== TRAIN DATA OVERVIEW ===")
    print("Columns:", train.columns.tolist())
    train['rpm'] = train['posted_rate'] / train['distance']

    # Let's inspect quote_signal vs rpm
    train['qs_error'] = train['rpm'] - train['quote_signal']
    print("Quote signal vs actual RPM:")
    print("Mean abs error between quote_signal and rpm:", np.abs(train['qs_error']).mean())
    print("Median abs error:", np.median(np.abs(train['qs_error'])))

    # Does quote_signal correlate with distance, equipment, etc?
    print("\nQuote signal correlation with features:")
    print(train[['distance', 'weight', 'market_index', 'quote_signal', 'rpm']].corr())

    # Check December chart inputs:
    print("\nDecember chart inputs:")
    print(dec.head(5))

    # What happens if we predict December?
    # December has:
    # pickup: Lexington, delivery: Fort Wayne, distance: 360, equipment: Dry Van, weight: 32000, date: 2025-12-01 to 2025-12-31
    # Can we get quote_signal for December? December input CSV has NO quote_signal!
    # Does validation.csv have quote_signal? Yes!
    # What if a model uses quote_signal? If a model used quote_signal, how could it predict December?
    # It couldn't predict December unless quote_signal was imputed or predicted!
    # OR, is quote_signal supposed to be used or NOT used?
    # Let's check prompt Step 1:
    # 'Look for data-quality issues: inconsistent city names or casing, whitespace, impossible distances/weights, rate outliers, rate-per-mile anomalies, duplicate rows, mixed date formats, leakage columns. Decide on and document the handling of each (fix, clip, drop or flag), and keep the rules identical for train and validation.'
    # And Step 3:
    # 'Features: rate-per-mile style features derived only from training data (no target leakage), distance, weight, equipment, lane (pickup-delivery) with target encoding done inside CV folds, date features (day of week, month, day of month, week of year, holiday proximity, especially for December), and a trend/recency feature.'

    print("\nCheck if quote_signal is in validation.csv:")
    print("val columns:", val.columns.tolist())
    print("dec columns:", dec.columns.tolist())

if __name__ == '__main__':
    run()
