# -*- mode: python ; coding: utf-8 -*-

import os

code_dir = os.environ['MYMI_CODE']
data_dir = os.environ['MYMI_DATA']

localiser = os.path.join('models', 'localiser-Brain', 'public-1gpu-150epochs')
localiser_ckpt = 'loss=-0.957796-epoch=148-step=0.ckpt'
segmenter = os.path.join('models', 'segmenter-replan-eyes', '4-regions-EL_ER_LL_LR-112-seed-43-ivw-1.0-schedule-200-lr-1e-4')
segmenter_alias = os.path.join('models', 'segmenter', 'eyes')
segmenter_ckpt = 'loss=-0.153282-epoch=675-step=0.ckpt'

datas = [
    # It seems that when 'from lightning import __version__' is called from the submodule 'lightning-fabric'
    # 'pyinstaller' has copied the 'lighting' version.py file to 'lightning-fabric' version.py and so
    # the paths are messed up when looking for 'version.info'.
    ('/home/baclark/code/lightning/src/lightning/version.info', 'lightning_fabric'),
    ('/home/baclark/code/lightning/src/pytorch_lightning/__version__.py', 'pytorch_lightning'),
    ('/home/baclark/code/lightning/src/pytorch_lightning/version.info', 'pytorch_lightning'),
    (os.path.join(data_dir, localiser, localiser_ckpt), localiser),
    (os.path.join(data_dir, segmenter, segmenter_ckpt), segmenter_alias)
]
hiddenimports = [
    'pydicom.encoders.gdcm',
    'pydicom.encoders.pylibjpeg',
    'pydicom.encoders.native',
    'wandb_gql'
]
hookspath = [
    os.path.join(code_dir, 'scripts', 'predict', 'dicom', 'segmenter', 'extra-hooks')
]
pathex = [
    code_dir,
    # Add 'code' path as we have cloned and altered some repos (e.g. pytorch_lightning).
    '/home/baclark/code'
]

a = Analysis(
    ['predict_eyes.py'],
    binaries=[],
    datas=datas,
    excludes=[],
    hiddenimports=hiddenimports,
    hookspath=hookspath,
    hooksconfig={},
    noarchive=False,
    pathex=pathex,
    runtime_hooks=[],
)

print(f'orig datas: {datas}')
print(f'orig hiddenimports: {hiddenimports}')
print(f'orig hookspath: {hookspath}')

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='predict_eyes',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='predict_eyes',
)
