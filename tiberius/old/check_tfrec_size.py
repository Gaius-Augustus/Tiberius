import argparse
import sys
from tiberius import DataGenerator


def check_tfrecord_sequence_length(tfrecord_path, expected_length):
    """
    Check if all examples in a TFRecord file have sequences of the expected length in the first dimension.

    Args:
        tfrecord_path (str): Path to the TFRecord file.
        expected_length (int): Expected length of the sequences in the first dimension.

    Returns:
        bool: True if all sequences have the expected length, False otherwise.
    """

    generator = DataGenerator(file_path=[tfrecord_path], 
          batch_size=1, 
          shuffle=False,
          repeat=False,
          filter=False,
          output_size=15,
          hmm_factor=0,
          clamsa=False ,
          oracle=False 
      )
    
    # r = generator.__next__()
    # print(r.shape[0])
    for record in generator: 
        if record[0][0].shape[0] != expected_length or record[1][0].shape[0] != expected_length:
            print(record[0].shape, record[1].shape)
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Check sequence lengths in a TFRecord file.")
    parser.add_argument("--tfrecord_path", type=str, required=True, help="Path to the TFRecord file.")
    parser.add_argument("--expected_length", type=int, default=9999, help="Expected length of the sequences in the first dimension.")
    args = parser.parse_args()

    result = check_tfrecord_sequence_length(args.tfrecord_path, args.expected_length)
    if result:
        print("All sequences have the correct length.")
    else:
        print("Some sequences have incorrect lengths.")

if __name__ == "__main__":
    main()