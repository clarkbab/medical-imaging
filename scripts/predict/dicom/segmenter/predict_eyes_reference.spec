# -*- mode: python ; coding: utf-8 -*-

mymipath = "/data/projects/punim1413/medical-imaging"
localiser = "localiser-Brain/public-1gpu-150epochs/loss=-0.957796-epoch=148-step=0.ckpt"
segmenter = "segmenter-replan-eyes/4-regions-EL_ER_LL_LR-112-seed-42-lr-1e-4/loss=-0.096713-epoch=671-step=0.ckpt"

a = Analysis(
    ['predict_eyes.py'],
    pathex=[
        mymipath
    ],
    binaries=[],
    datas=[
        (f'/data/projects/punim1413/mymi/models/{localiser}', f'data/models/{localiser}'),
        (f'/data/projects/punim1413/mymi/models/{segmenter}', f'data/models/{segmenter}')
    ],
    hiddenimports=[
        'pydicom.encoders.gdcm'
    ],
    hookspath=[
        'extra-hooks'
    ],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
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
