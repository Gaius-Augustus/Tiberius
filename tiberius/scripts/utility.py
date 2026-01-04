import yaml, logging, os, shutil, subprocess, sys
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

#Authors: "Amrei Knuth", "Lars Gabriel"
#Credits: "Katharina Hoff"
#Email:"lars.gabriel@uni-greifswald.de"
#Date: "Janurary 2025"

logger = logging.getLogger(__name__)

def read_yaml(file_path):
    """Read a YAML file and return its contents as a dictionary."""
    with open(file_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def find_executable(tool_path, executable_name, tool_name=''):
    """
    Find the path to a tool's executable, checking user-specified paths and environment variables.

    :param tool_path: Path provided by user
    :param executable_name: The name of the executable to check (e.g., "hisat2").
    :return: The directory containing the executable.
    """
    # Determine the user-specified path (either from args or config)

    # Check the user-specified path
    if tool_path:
        executable_path = os.path.join(tool_path, executable_name)
        if os.access(executable_path, os.X_OK):
            tool_path = make_absolute_path(tool_path)
            logging.info(f"{executable_name} found in: {tool_path}")
            return tool_path

    # Check the system PATH
    executable_in_path = shutil.which(executable_name)
    if executable_in_path:
        tool_path = os.path.dirname(executable_in_path)
        tool_path = make_absolute_path(tool_path)
        logging.info(f"{executable_name} found in system PATH at: {tool_path}")
        return tool_path

    # If not found, print an error and exit
    logging.error(f"Error: {executable_name} executable wasn't found.")
    error_message = ''
    if tool_name:
        error_message = f"For {tool_name}"
    error_message += f"Please provide the path to {executable_name} in your config file or as command line argument."
    logging.error(error_message)
    raise ValueError(error_message)

# create dir if not exists
def make_directory(path):
    """
    Ensures that a directory exists at the specified path.
    If the directory does not exist, it is created.

    :param path: The path to the directory to check or create.
    """
    try:
        # Check if the path exists
        if not os.path.exists(path):
            # Create the directory
            os.makedirs(path)
    except Exception as e:
        print(f"Error creating directory at {path}: {e}")
        raise 

def make_absolute_path(path, create_if_missing=False):
    """
    Converts a directory path to its absolute path.
    Optionally creates the directory if it doesn't exist.

    :param path: The directory path to convert.
    :param create_if_missing: Whether to create the directory if it doesn't exist.
    :return: The absolute path of the directory.
    :raises FileNotFoundError: If the directory doesn't exist and create_if_missing is False.
    """
    # Convert to absolute path
    absolute_path = os.path.abspath(path)

    # Check if the directory exists
    if not os.path.exists(absolute_path):
        if create_if_missing:
            os.makedirs(absolute_path)
            print(f"Directory created at: {absolute_path}")
        else:
            raise FileNotFoundError(f"Directory does not exist: {absolute_path}")
    return absolute_path


def file_format(file):
    """
    Determine the format of a sequence file (FASTA, FASTQ, or unknown).

    This function reads the first few lines of a file to identify its format based on
    common markers in FASTA and FASTQ files:
    - FASTA files start with a '>' character.
    - FASTQ files start with a '@' character, and the third line starts with a '+' character.

    :param file: Path to the file whose format needs to be determined.
    :return: A string indicating the file format: 'fasta', 'fastq', or 'unknown'.
    :raises FileNotFoundError: If the specified file does not exist.
    :raises IOError: If the file cannot be read for any reason.
    """
    try:
        with open(file, 'r') as f:
            # Read the first three lines
            first_line = f.readline().strip()
            second_line = f.readline().strip()
            third_line = f.readline().strip()

        # Identify the format
        if first_line.startswith('>'):
            return "fasta"
        elif first_line.startswith('@') and third_line.startswith('+'):
            return "fastq"
        else:
            return "unknown"
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: File not found - {file}")
    except IOError as e:
        raise IOError(f"Error reading file {file}: {e}")

def run_subprocess(command, capture_output=True, text=True, 
            check=True, error_message="Subprocess execution failed",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, stdin=None):
    """
    Execute a subprocess command with error handling and logging.

    This function executes a given command using `subprocess.run`, captures its output,
    and handles errors gracefully. If the command fails, it logs the error and raises an exception.

    :param command: List of command arguments to execute (e.g., ["ls", "-l"]).
    :param capture_output: Whether to capture the command's stdout and stderr (default: True).
    :param text: If True, returns stdout and stderr as strings instead of bytes (default: True).
    :param check: If True, raises a `subprocess.CalledProcessError` on a non-zero return code (default: True).
    :param error_message: Custom error message to log and raise if the command fails (default: "Subprocess execution failed").
    :return: The result of the subprocess (subprocess.CompletedProcess object).
    :raises RuntimeError: If the subprocess fails and `check` is set to False.
    :raises subprocess.CalledProcessError: If the subprocess fails and `check` is set to True.
    """
    try:
        logging.info(f"Executing command: {' '.join(command)}")
        if capture_output:
            result = subprocess.run(command, capture_output=capture_output, text=text, 
                            check=check, shell=shell, stdin=stdin)
        else:
            result = subprocess.run(command, capture_output=capture_output, text=text, check=check,
                            stdout=stdout, stderr=stderr, shell=shell, stdin=stdin)
        if result.returncode == 0:
            logging.info("Subprocess completed successfully.")
        else:
            logging.warning(f"Subprocess completed with non-zero return code: {result.returncode}")

        return result

    except subprocess.CalledProcessError as e:
        logging.error(f"{error_message}: {e}")
        logging.error(f"Command: {' '.join(command)}")
        logging.error(f"stdout: {e.stdout}")
        logging.error(f"stderr: {e.stderr}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error while running subprocess: {e}")
        logging.error(f"Command: {' '.join(command)}")
        raise RuntimeError(f"{error_message}: {e}") from e


def merge_bam_files(bam_files, samtools_path, 
                    output_bam="alignment_merged_rnaseq.bam"):
    """
    Merge two BAM files into a single BAM file using SAMtools.

    This function merges two BAM files into a single output BAM file using the `samtools merge` command. 
    The merged BAM file is saved with a default name, `alignment_merged_rnaseq.bam`.

    :param bam_files: List of bam files.
    :param samtools_path: Path to the directory containing the SAMtools executable.
    :return: The path to the merged BAM file.
    """    
    if not bam_files:
        raise ValueError("The list of BAM files to merge is empty.")

    samtools_executable = os.path.join(samtools_path, "samtools")
    
    # Define the output BAM file
    output_bam = "alignment_merged_rnaseq.bam"

    # Construct the SAMtools merge command
    command = [
        samtools_executable,
        "merge",
        "-f",  # Overwrite the output file if it exists
        "-o", output_bam,
        *bam_files 
    ]

    logging.info(f"Merging BAM files: {bamfile_1} and {bamfile_2} into {output_bam} ...")
    error_msg  = f"SAMtools merge failed for {bamfile_1},{bamfile_2}. Check logs for details."
    run_subprocess(command, error_message=error_msg)
    logging.info(f"Merged BAM files successfully into: {output_bam} .")

    return output_bam


def sam_to_bam(sam_file_list, samtools_path, threads=4, merge_bam='', use_existing=False):    
    """Convert SAM files to BAM files using SAMtools.

    :param sam_file_list: List of SAM files to convert to BAM format.
    :param samtools_path: Path to the SAMtools executable directory.
    :param threads: Number of threads to use for sorting (default: 4).
    :param merge_bam: Path to merged BAM file of input SAM file. (default no merging)).
    :return: None
    """
    # Path to the SAMtools executable
    samtools_executable = os.path.join(samtools_path, "samtools")
    output_bam_list = []

    for samfile in sam_file_list:
        # Generate output BAM file name
        output_bam = os.path.splitext(samfile)[0] + ".bam"
        command = [
            samtools_executable,
            "sort",
            "-@",
            str(threads),
            samfile,
            "-o",
            output_bam
        ]
        logging.info(f"Converting {samfile} to {output_bam}...")
        if use_existing and file_exists_and_not_empty(output_bam):
            logging.info(f"Using existing file instead of rerunning program: {output_bam}")
        else:
            error_msg  = f"Error during conversion of {samfile}:"
            run_subprocess(command, error_message=error_msg)
            logging.info(f"Conversion from {samfile} to {output_bam} completed successfully.")
        output_bam_list.append(output_bam)

    if merge_bam:
        merge_bam_files(output_bam_list, samtools_path, output_bam=merge_bam)
        return merge_bam
    
    return output_bam_list

def convert_gtf_to_gff3(transcripts_gtf, transdecoder_path, output_gff3="output.gff3"):
    """
    Convert a GTF file to GFF3 format using the TransDecoder `gtf_to_alignment_gff3.pl` module.

    This function uses the TransDecoder `gtf_to_alignment_gff3.pl` script to convert a GTF file 
    into GFF3 format and writes the output to the specified directory and file.

    :param transcripts_gtf: Path to the input GTF file.
    :param transdecoder_path: Path to the directory containing the TransDecoder `gtf_to_alignment_gff3.pl` script.
    :param output_gff3: Path of the output GFF3 file (default: "output.gff3").
    :return: Path to the generated GFF3 file.
    """
    # Construct the path to the TransDecoder script
    transdecoder_script = os.path.join(transdecoder_path, "gtf_to_alignment_gff3.pl")

    # Construct the command
    command = [transdecoder_script, transcripts_gtf]

    # Log the process
    logging.info(f"Converting GTF file {transcripts_gtf} to GFF3 format...")
    logging.info(f"Output will be written to: {output_gff3}")

    # Run the subprocess
    try:
        with open(output_gff3, "w") as output_file:
            error_msg = f"TransDecoder conversion failed for {transcripts_gtf}. Check logs for details."
            run_subprocess(command, error_message=error_msg, stdout=output_file, capture_output=False)
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        raise

    logging.info(f"Conversion completed successfully. GFF3 file written to {output_gff3}")

    return output_gff3

def file_exists_and_not_empty(file_path):
    """
    Checks if a file exists and is not empty.

    :param file_path: Path to the file to check.
    :return: True if the file exists and is not empty, False otherwise.
    """
    try:
        # Check if the file exists
        if not os.path.isfile(file_path):
            return False

        # Check if the file is not empty
        if os.path.getsize(file_path) > 0:
            return True
        else:
            return False

    except Exception as e:
        raise RuntimeError(f"Error checking file: {file_path}. Details: {e}")


def copy_file(src, dest):
    """
    Copies a file from `src` to `dest`.

    - If `dest` is a directory, the file is copied inside it with the same name.
    - If `dest` is a file path, the file is copied with that name.
    - If `src` does not exist, an error is raised.
    """
    try:
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Source file '{src}' does not exist.")

        copied_path = shutil.copy2(src, dest)
        print(f"File copied successfully to '{copied_path}'")
        return copied_path

    except Exception as e:
        print(f"Error copying file: {e}")
        return None