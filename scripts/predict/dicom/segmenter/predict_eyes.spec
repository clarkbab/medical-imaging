# -*- mode: python ; coding: utf-8 -*-

datas = [
    # It seems that when 'from lightning import __version__' is called from the submodule 'lightning-fabric'
    # 'pyinstaller' has copied the 'lighting' version.py file to 'lightning-fabric' version.py and so
    # the paths are messed up when looking for 'version.info'.
    ('/home/baclark/code/lightning/src/lightning/version.info', 'lightning_fabric')
]
hiddenimports = [
    'pydicom.encoders.gdcm',
    'pydicom.encoders.pylibjpeg',
    'pydicom.encoders.native',
    'wandb_gql'
]
hookspath = [
    '/data/projects/punim1413/medical-imaging/scripts/predict/dicom/segmenter/extra-hooks'
]
pathex = [
    '/data/projects/punim1413/medical-imaging',
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
