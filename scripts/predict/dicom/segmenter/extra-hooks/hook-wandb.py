# From: https://github.com/pyinstaller/pyinstaller/issues/7655

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

print('executing wandb hook')

datas = collect_data_files('wandb', include_py_files=True, includes=['**/vendor/**/*.py'])

hiddenimports = collect_submodules('pathtools')

print(f'datas: {datas}')
print(f'hiddenimports: {hiddenimports}')
