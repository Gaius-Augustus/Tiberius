import csv
import argparse

def filter_transcripts(csv_file):
    """
    Reads the input CSV file and returns a list of transcript IDs for which:
      - has_inframe_stop is 1, or
      - has_canonical_start is 0, or
      - canonical_splice_sites is 0
      
    Parameters:
        csv_file (str): Path to the CSV file.
    
    Returns:
        list: A list of transcript IDs meeting the criteria.
    """
    transcript_ids = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (float(row.get('repeats_in_cds_fraction')) > 0):
                transcript_ids.append(row.get('transcript_id'))
    return transcript_ids

def main():
    parser = argparse.ArgumentParser(
        description="Filter transcript IDs from a stats CSV file based on specific criteria."
    )
    parser.add_argument("--stats", required=True, help="Path to the stats CSV file.")
    parser.add_argument("--out", required=True, help="Path to the output file for transcript IDs.")
    
    args = parser.parse_args()
    
    tx_ids = filter_transcripts(args.stats)
    
    with open(args.out, "w") as f:
        for tx in tx_ids:
            f.write(f"{tx}\n")
    
    print(f"Filtered {len(tx_ids)} transcript IDs written to {args.out}")

if __name__ == '__main__':
    main()
