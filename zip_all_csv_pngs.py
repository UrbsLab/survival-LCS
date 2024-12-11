import os
import zipfile

def zip_csv_files(root_dir, output_zip):
    """
    Recursively gather and zip all CSV files from a directory.

    :param root_dir: The root directory to search for CSV files.
    :param output_zip: The output ZIP file name.
    """
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.csv') or file.endswith('.png'):
                    filepath = os.path.join(dirpath, file)
                    arcname = os.path.relpath(filepath, root_dir)
                    zipf.write(filepath, arcname)
    print(f"CSV files successfully zipped into {output_zip}")

# Example usage
root_directory = "./pipeline"
output_zip_file = "./output/pipeline-gamma-files.zip"

zip_csv_files(root_directory, output_zip_file)
