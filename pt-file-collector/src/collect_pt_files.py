import os
import shutil
import argparse

def collect_pt_files(source_dir, target_dir):
    """Collect all .pt files from the source directory and move them to the target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.pt'):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_dir, file)
                shutil.move(source_file_path, target_file_path)
                print(f'Moved: {source_file_path} to {target_file_path}')

def main():
    parser = argparse.ArgumentParser(description='Collect all .pt files from a directory.')
    parser.add_argument('source_dir', type=str, help='The source directory to search for .pt files.')
    parser.add_argument('target_dir', type=str, help='The target directory to move .pt files to.')
    args = parser.parse_args()

    collect_pt_files(args.source_dir, args.target_dir)

if __name__ == '__main__':
    main()