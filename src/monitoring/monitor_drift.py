import pandas as pd
from evidently.report import Report
# Use the specific metrics for a more targeted and robust test
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ColumnDriftMetric
)
import os

# --- Config ---
REFERENCE_FILE = "data/reference_data.jsonl"
CURRENT_LOGS_FILE = "data/logs/prediction_logs.jsonl"
DRIFT_REPORT_FILE = "data/reports/drift_report.html"

def load_and_normalize(ref_path, cur_path):
    """
    Loads both reference and current data, and normalizes them
    to have the exact same set of columns. This is the most
    robust way to compare them.
    """
    try:
        ref_df = pd.read_json(ref_path, lines=True)
        cur_df = pd.read_json(cur_path, lines=True)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return pd.DataFrame(), pd.DataFrame()
    except ValueError as e:
        print(f"Warning: A log file is empty. {e}")
        return pd.DataFrame(), pd.DataFrame() # Return empty DFs

    if ref_df.empty or cur_df.empty:
        return ref_df, cur_df

    # Flatten the nested JSON into columns
    ref_flat = pd.json_normalize(ref_df['class_distribution']).add_prefix('class_')
    cur_flat = pd.json_normalize(cur_df['class_distribution']).add_prefix('class_')
    
    # Combine with main DFs
    ref_df = pd.concat([ref_df.drop('class_distribution', axis=1), ref_flat], axis=1)
    cur_df = pd.concat([cur_df.drop('class_distribution', axis=1), cur_flat], axis=1)
    
    # --- Robust Column Normalization Logic ---
    # Find all unique columns from both dataframes
    all_cols = set(ref_df.columns).union(set(cur_df.columns))
    
    # Re-index both dataframes to have the same columns, filling missing with 0
    # This prevents errors and false negatives during drift detection.
    ref_df = ref_df.reindex(columns=all_cols, fill_value=0)
    cur_df = cur_df.reindex(columns=all_cols, fill_value=0)
    
    return ref_df, cur_df

def main():
    print("Starting drift monitoring...")
    
    # 1. Load and normalize reference and current data
    reference_df, current_df = load_and_normalize(REFERENCE_FILE, CURRENT_LOGS_FILE)

    if reference_df.empty:
        print(f"Error: Reference data at {REFERENCE_FILE} is empty. Run generate_reference.py first.")
        return
    if current_df.empty:
        print(f"No current prediction logs found at {CURRENT_LOGS_FILE}. Stopping.")
        return

    # 2. Create a more SPECIFIC drift report
    print("Running Evidently AI report...")
    data_drift_report = Report(metrics=[
        DatasetDriftMetric(), # This gives us the overall True/False
        DataDriftTable(),     # This creates the big table of all columns
        
        # --- Add specific tests for the columns we care about ---
        # This ensures we catch drift even if other columns mask it
        ColumnDriftMetric(column_name="avg_confidence"),
        ColumnDriftMetric(column_name="num_boxes_predicted"),
    ])

    # Remove the 'column_mapping' argument for broader compatibility
    data_drift_report.run(current_data=current_df, 
                            reference_data=reference_df)
    
    # 3. Save the HTML report
    data_drift_report.save_html(DRIFT_REPORT_FILE)
    print(f"Drift report saved to {DRIFT_REPORT_FILE}")

    # 4. Check for drift and alert
    drift_results = data_drift_report.as_dict()
    # This path points to the result of the DatasetDriftMetric()
    drift_detected = drift_results['metrics'][0]['result']['dataset_drift']
    
    if drift_detected:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!      DATA DRIFT DETECTED     !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        print("No data drift detected. All systems nominal.")

if __name__ == "__main__":
    main()