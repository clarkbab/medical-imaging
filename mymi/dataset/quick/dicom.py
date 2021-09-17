import dicom2nifti as d2n

def convert_dicom_to_nifti(
    path: str,
    output_path: str) -> None:
    d2n.convert_directory(path, output_path, compression=True, reorient=True)
